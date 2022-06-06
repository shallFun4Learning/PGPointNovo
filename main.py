
import numpy as np
import random
import torch
import argparse
from torch._C import wait
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import subprocess
import cProfile
import logging
import logging.config
import config
# import train_func
import time
from data_reader import DeepNovoDenovoDataset, collate_func, DeepNovoTrainDataset, DBSearchDataset, denovo_collate_func, del_file
from db_searcher import DataBaseSearcher
from denovo import IonCNNDenovo
from writer import DenovoWriter, PercolatorWriter
import deepnovo_worker_test
from psm_ranker import PSMRank
from deepnovo_dia_script_select import find_score_cutoff
import os
import shutil
from train_func import train, build_model, validation, perplexity
from model import InferenceModelWrapper

logger = logging.getLogger(__name__)


dist.init_process_group(backend='nccl')
local_rank = config.local_rank
print(
    '/'*30+f'now is ({local_rank}), cofig.local_rank = {config.local_rank} '+'/'*30)
torch.cuda.set_device(local_rank)
print('/'*30+f'torch.cuda.set_device({local_rank})'+'/'*30)


def init_seeds(seed=825):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    if config.FLAGS.train:
        dist.barrier()
        if dist.get_rank() == 0:
            if os.path.exists(config.train_dir):
                shutil.move('train', 'history_train'+os.sep+config.dataset_name +
                            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
            assert not os.path.exists('train'), ' there is train'
            os.makedirs(config.train_dir)
        dist.barrier()
        s = time.time()
        logger.info("training mode")
        train()
        e = time.time()
        with open('results/rec', "a+") as rec:
            print('*'*30, 'Tarin Time ', '*'*30, file=rec)
            print(f'num_workers: {config.num_workers}', file=rec)
            print(f'Total time: {e-s} sec, {(e-s)/3600} h .', file=rec)
    elif config.FLAGS.search_denovo:

        s = time.time()
        logger.info("denovo mode")
        data_reader = DeepNovoDenovoDataset(feature_filename=config.denovo_input_feature_file,
                                            spectrum_filename=config.denovo_input_spectrum_file)
        data_sampler = torch.utils.data.distributed.DistributedSampler(
            data_reader, shuffle=False)
        denovo_data_loader = torch.utils.data.DataLoader(dataset=data_reader, batch_size=config.batch_size,
                                                         #  shuffle=False,
                                                         sampler=data_sampler,
                                                         num_workers=config.num_workers,
                                                         collate_fn=denovo_collate_func)
        denovo_worker = IonCNNDenovo(config.MZ_MAX,
                                     config.knapsack_file,
                                     beam_size=config.FLAGS.beam_size)

        forward_deepnovo, backward_deepnovo, init_net = build_model(
            training=False)

        model_wrapper = InferenceModelWrapper(
            forward_deepnovo, backward_deepnovo, init_net)
        if dist.get_rank() == 0:
            del_file(config.dataset_name+os.sep+'denovo')
        dist.barrier()
        denovo_path = config.dataset_name+os.sep+'denovo'+os.sep + \
            config.denovo_output_file.split('.')[-1]+str(local_rank)
        writer = DenovoWriter(denovo_path)

        with torch.no_grad():
            denovo_worker.search_denovo(
                model_wrapper, denovo_data_loader, writer)
            # cProfile.runctx("denovo_worker.search_denovo(model_wrapper, denovo_data_loader, writer)", globals(), locals())
        e = time.time()
        logger.info(
            f"de novo {len(data_reader)} spectra takes {time.time() - s} seconds")
        dist.barrier()
        print(f'GPU : {dist.get_rank()} has completed denovo tasks. ')

    elif config.FLAGS.valid:
        valid_set = DeepNovoTrainDataset(config.input_feature_file_valid,
                                         config.input_spectrum_file_valid)
        valid_sampler = torch.utils.data.distributed.DistributedSampler(
            valid_set, shuffle=False)
        valid_data_loader = torch.utils.data.DataLoader(dataset=valid_set,
                                                        batch_size=config.batch_size,
                                                        # shuffle=False,
                                                        sampler=valid_sampler,
                                                        num_workers=config.num_workers,
                                                        collate_fn=collate_func)
        forward_deepnovo, backward_deepnovo, init_net = build_model(
            training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()
        validation_loss = validation(
            forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)
        logger.info(f"validation perplexity: {perplexity(validation_loss)}")

    elif config.FLAGS.test:
        if dist.get_rank() == 0:
            from data_reader import merge
            merge()
            logger.info("test mode")
            worker_test = deepnovo_worker_test.WorkerTest()
            worker_test.test_accuracy()

            # show 95 accuracy score threshold
            accuracy_cutoff = 0.95
            accuracy_file = config.accuracy_file
            score_cutoff = find_score_cutoff(accuracy_file, accuracy_cutoff)

    elif config.FLAGS.search_db:
        logger.info("data base search mode")
        start_time = time.time()
        db_searcher = DataBaseSearcher(config.db_fasta_file)
        dataset = DBSearchDataset(config.search_db_input_feature_file,
                                  config.search_db_input_spectrum_file,
                                  db_searcher)
        num_spectra = len(dataset)

        def simple_collate_func(train_data_list):
            return train_data_list

        data_reader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=1,
                                                  shuffle=False,
                                                  num_workers=config.num_db_searcher_worker,
                                                  collate_fn=simple_collate_func)

        forward_deepnovo, backward_deepnovo, init_net = build_model(
            training=False)
        forward_deepnovo.eval()
        backward_deepnovo.eval()

        writer = PercolatorWriter(config.db_output_file)
        psm_ranker = PSMRank(data_reader, forward_deepnovo,
                             backward_deepnovo, writer, num_spectra)
        psm_ranker.search()
        writer.close()
        # call percolator
        with open(f"{config.db_output_file}" + '.psms', "w") as fw:
            subprocess.run(["percolator", "-X", "/tmp/pout.xml", f"{config.db_output_file}"],
                           stdout=fw)

    else:
        raise RuntimeError("unspecified mode")


if __name__ == '__main__':
    log_file_name = 'PointNovo.log'
    d = {
        'version': 1,
        'disable_existing_loggers': False,  # this fixes the problem
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'console': {
                'level': 'INFO',
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.FileHandler',
                'filename': log_file_name,
                'mode': 'w',
                'formatter': 'standard',
            }
        },
        'root': {
            'handlers': ['console', 'file'],
            'level': 'DEBUG',
        }
    }
    logging.config.dictConfig(d)
    main()
    # CUDA_VISIBLE_DEVICES="0,2,4,5,6" python -m torch.distributed.launch --nproc_per_node 5 main.py
