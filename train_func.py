import time
from itertools import count
from torch.distributed.distributed_c10d import gather, get_rank
from model import DeepNovoModel, device, InitNet
import model
import torch.distributed as dist
import config
import os
import torch
from torch import optim, nn
import torch.nn.functional as F
from data_reader import DeepNovoTrainDataset, collate_func
import pandas as pd
import time
import math
import logging
from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed.optim import ZeroRedundancyOptimizer
from ranger2020 import Ranger
from torch.utils.tensorboard import SummaryWriter
logger = logging.getLogger(__name__)

forward_model_save_name = 'forward_deepnovo.pth'
backward_model_save_name = 'backward_deepnovo.pth'
init_net_save_name = 'init_net.pth'
device = torch.device("cuda",  config.local_rank)


def to_one_hot(y, n_dims=None):
    """ Take integer y with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(
        y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return y_one_hot


def focal_loss(logits, labels, ignore_index=-100, gamma=2.):
    """
    :param logits: float tensor of shape [batch, T, 26]
    :param labels: long tensor of shape [batch, T]
    :param ignore_index: ignore the loss of those tokens
    :param gamma:
    :return: average loss, num_valid_token
    """
    valid_token_mask = (labels != ignore_index).float().to(
        device)  # [batch, T]
    num_valid_token = torch.sum(valid_token_mask).to(device)
    batch_size, T, num_classes = logits.size()
    sigmoid_p = torch.sigmoid(logits).to(device)
    target_tensor = to_one_hot(labels, n_dims=num_classes).float().to(device)
    zeros = torch.zeros_like(sigmoid_p).to(device)
    pos_p_sub = torch.where(target_tensor >= sigmoid_p,
                            target_tensor - sigmoid_p, zeros).to(device)  # [batch, T, 26]
    neg_p_sub = torch.where(target_tensor > zeros, zeros,
                            sigmoid_p).to(device)  # [batch, T, 26]

    per_token_loss = - (pos_p_sub ** gamma) * torch.log(torch.clamp(sigmoid_p, 1e-8, 1.0)) - \
        (neg_p_sub ** gamma) * \
        torch.log(torch.clamp(1.0 - sigmoid_p, 1e-8, 1.0)).to(device)
    per_entry_loss = torch.sum(per_token_loss, dim=2).to(device)  # [batch, T]
    # masking out loss from pad tokens
    per_entry_loss = per_entry_loss * valid_token_mask

    per_entry_average_loss = torch.sum(
        per_entry_loss) / (num_valid_token + 1e-6).to(device)
    return per_entry_average_loss, num_valid_token


def build_model(training=True):
    """

    :return:
    """
    # device = torch.device("cuda", local_rank)
    forward_deepnovo = DeepNovoModel()
    backward_deepnovo = DeepNovoModel()
    if config.use_lstm:
        init_net = InitNet()
    else:
        init_net = None

    # load pretrained params if exist
    if os.path.exists(os.path.join(config.train_dir, forward_model_save_name)):
        assert os.path.exists(os.path.join(
            config.train_dir, backward_model_save_name))
        logger.info("load pretrained model")
        forward_deepnovo.load_state_dict(torch.load(os.path.join(config.train_dir, forward_model_save_name),
                                                    map_location=device))
        backward_deepnovo.load_state_dict(torch.load(os.path.join(config.train_dir, backward_model_save_name),
                                                     map_location=device))
        if config.use_lstm:
            init_net.load_state_dict(torch.load(os.path.join(config.train_dir, init_net_save_name),
                                                map_location=device))
    else:
        assert training, f"building model for testing, but could not found weight under directory " \
                         f"{config.train_dir}"
        logger.info("initialize a set of new parameters")

    if config.use_lstm:
        # share embedding matrix
        backward_deepnovo.embedding.weight = forward_deepnovo.embedding.weight

    # backward_deepnovo = backward_deepnovo.to(device)
    backward_deepnovo = backward_deepnovo.to(device)
    forward_deepnovo = forward_deepnovo.to(device)

    if config.use_sync:
        backward_deepnovo = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            backward_deepnovo).to(device)
        forward_deepnovo = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            forward_deepnovo).to(device)
        if config.use_lstm:
            # init_net = init_net.to(device)
            init_net = init_net.to(device)
            init_net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                init_net).to(device)

    backward_deepnovo = DDP(backward_deepnovo, device_ids=[
                            config.local_rank], output_device=config.local_rank)
    # print('/'*10+f'device_ids=[{local_rank}]'+'/'*10)
    # forward_deepnovo = forward_deepnovo.to(device)

    forward_deepnovo = DDP(forward_deepnovo, device_ids=[
        config.local_rank], output_device=config.local_rank)
    if config.use_lstm:
        init_net = DDP(init_net, device_ids=[
            config.local_rank], output_device=config.local_rank)
    return forward_deepnovo, backward_deepnovo, init_net


def extract_and_move_data(data):
    """

    :param data: result from dataloader
    :return:
    """
    peak_location, \
        peak_intensity, \
        spectrum_representation,\
        batch_forward_id_target, \
        batch_backward_id_target, \
        batch_forward_ion_index, \
        batch_backward_ion_index, \
        batch_forward_id_input, \
        batch_backward_id_input = data

    # move to device
    peak_location = peak_location.to(device)
    peak_intensity = peak_intensity.to(device)
    spectrum_representation = spectrum_representation.to(device)
    batch_forward_id_target = batch_forward_id_target.to(device)
    batch_backward_id_target = batch_backward_id_target.to(device)
    batch_forward_ion_index = batch_forward_ion_index.to(device)
    batch_backward_ion_index = batch_backward_ion_index.to(device)
    batch_forward_id_input = batch_forward_id_input.to(device)
    batch_backward_id_input = batch_backward_id_input.to(device)
    return (peak_location,
            peak_intensity,
            spectrum_representation,
            batch_forward_id_target,
            batch_backward_id_target,
            batch_forward_ion_index,
            batch_backward_ion_index,
            batch_forward_id_input,
            batch_backward_id_input
            )


def validation(forward_deepnovo, backward_deepnovo, init_net, valid_loader) -> float:
    with torch.no_grad():
        valid_loss = 0
        num_valid_samples = 0
        for data in valid_loader:
            peak_location, \
                peak_intensity, \
                spectrum_representation, \
                batch_forward_id_target, \
                batch_backward_id_target, \
                batch_forward_ion_index, \
                batch_backward_ion_index, \
                batch_forward_id_input, \
                batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)
            if config.use_lstm:
                initial_state_tuple = init_net(spectrum_representation)
                forward_logit, _ = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity,
                                                    batch_forward_id_input, initial_state_tuple)
                backward_logit, _ = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity,
                                                      batch_backward_id_input, initial_state_tuple)
            else:
                forward_logit = forward_deepnovo(
                    batch_forward_ion_index, peak_location, peak_intensity)
                backward_logit = backward_deepnovo(
                    batch_backward_ion_index, peak_location, peak_intensity)
            forward_loss, f_num = focal_loss(
                forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, b_num = focal_loss(
                backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            valid_loss += forward_loss.item() * f_num.item() + \
                backward_loss.item() * b_num.item()
            num_valid_samples += f_num.item() + b_num.item()
    average_valid_loss = valid_loss / (num_valid_samples + 1e-6)
    return float(average_valid_loss)


def perplexity(log_loss):
    return math.exp(log_loss) if log_loss < 300 else float('inf')


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 3 epochs"""
    lr = config.init_lr * (0.1 ** ((epoch + 1) // 3))
    logger.info(f"epoch: {epoch}\tlr: {lr}")
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def save_model(forward_deepnovo, backward_deepnovo, init_net):
    torch.save(forward_deepnovo.module.state_dict(), os.path.join(config.train_dir,
                                                                  forward_model_save_name))
    torch.save(backward_deepnovo.module.state_dict(), os.path.join(config.train_dir,
                                                                   backward_model_save_name))
    if config.use_lstm:
        torch.save(init_net.module.state_dict(), os.path.join(config.train_dir,
                                                              init_net_save_name))


def train():
    dist.barrier()
    if dist.get_rank() == 0:
        writer = SummaryWriter()

    forward_deepnovo, backward_deepnovo, init_net = build_model()
    # sparse_params = forward_deepnovo.spectrum_embedding_matrix.parameters()

    dense_params = list(forward_deepnovo.parameters()) + \
        list(backward_deepnovo.parameters())

    if config.use_ranger:
        dense_optimizer = Ranger(
            dense_params, lr=config.init_lr, weight_decay=config.weight_decay)

    else:
        dense_optimizer = optim.Adam(dense_params,
                                     lr=config.init_lr,
                                     weight_decay=config.weight_decay)

    dense_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(dense_optimizer,
                                                                 'min',
                                                                 factor=config.factor,
                                                                 verbose=True,
                                                                 threshold=1e-4,
                                                                 cooldown=10,
                                                                 min_lr=config.init_lr*(1-config.factor))

    train_set = DeepNovoTrainDataset(config.input_feature_file_train,
                                     config.input_spectrum_file_train)

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, shuffle=True)
    train_data_loader = torch.utils.data.DataLoader(dataset=train_set,
                                                    batch_size=config.batch_size,
                                                    # shuffle=True,
                                                    sampler=train_sampler,
                                                    num_workers=config.num_workers,
                                                    collate_fn=collate_func)
    num_train_features = len(train_set)
    steps_per_epoch = int(num_train_features /
                          config.batch_size/config.num_GPUs)

    logger.info(f"{steps_per_epoch} steps per epoch")

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
    best_valid_loss = float("inf")

    # train loop

    best_epoch = None
    best_step = None
    start_time = time.time()
    valid_sampler.set_epoch(0)
    count = 0
    dist.barrier()
    totalstep = 0
    for epoch in range(config.num_epoch):
        train_sampler.set_epoch(epoch)
        # maybe there will be a warning,but i am sure that i want to do it.
        for i, data in enumerate(train_data_loader):
            totalstep += 1
            dense_optimizer.zero_grad()
            peak_location, \
                peak_intensity, \
                spectrum_representation, \
                batch_forward_id_target, \
                batch_backward_id_target, \
                batch_forward_ion_index, \
                batch_backward_ion_index, \
                batch_forward_id_input, \
                batch_backward_id_input = extract_and_move_data(data)
            batch_size = batch_backward_id_target.size(0)

            if config.use_lstm:
                initial_state_tuple = init_net(spectrum_representation)
                forward_logit, _ = forward_deepnovo(batch_forward_ion_index, peak_location, peak_intensity,
                                                    batch_forward_id_input, initial_state_tuple)
                backward_logit, _ = backward_deepnovo(batch_backward_ion_index, peak_location, peak_intensity,
                                                      batch_backward_id_input, initial_state_tuple)

            else:
                forward_logit = forward_deepnovo(
                    batch_forward_ion_index, peak_location, peak_intensity)
                backward_logit = backward_deepnovo(
                    batch_backward_ion_index, peak_location, peak_intensity)

            forward_loss, _ = focal_loss(
                forward_logit, batch_forward_id_target, ignore_index=0, gamma=2.)
            backward_loss, _ = focal_loss(
                backward_logit, batch_backward_id_target, ignore_index=0, gamma=2.)
            total_loss = (forward_loss + backward_loss) / 2.
            # compute gradient
            total_loss.backward()
            # clip gradient
            # torch.nn.utils.clip_grad_norm_(dense_params, deepnovo_config.max_gradient_norm)
            dense_optimizer.step()
            # sparse_optimizer.step()
            # torch.distributed.barrier()
            if dist.get_rank() == 0:
                writer.add_scalar('TrainLoss', total_loss, epoch)

            if (i + 1) % config.steps_per_validation == 0:
                count += 1

                duration = time.time() - start_time
                step_time = duration / config.steps_per_validation
                loss_cpu = total_loss.item()
                # evaluation mode
                forward_deepnovo.eval()
                backward_deepnovo.eval()
                validation_loss = validation(
                    forward_deepnovo, backward_deepnovo, init_net, valid_data_loader)

                # with open('temp/vl.temp', 'a+') as f_vl:
                #     print(
                #         f'On {dist.get_rank()} Gpus,  validation_loss={validation_loss}', file=f_vl)

                validation_loss = torch.tensor(
                    [validation_loss], dtype=torch.float32, device=model.device, requires_grad=False)

                dist.all_reduce(validation_loss,
                                op=torch.distributed.ReduceOp.SUM)
                validation_loss = float(validation_loss/config.num_GPUs)
                dense_scheduler.step(validation_loss)

                # with open('temp/vl.temp', 'a+') as f_vl:
                #     print(
                #         f'On {dist.get_rank()} Gpus,  get validation_loss={validation_loss}', file=f_vl)

                dist.barrier()
                if validation_loss < best_valid_loss:
                    best_valid_loss = validation_loss
                    logger.info(
                        f"best valid loss achieved at epoch {epoch} step {i}")
                    best_epoch = epoch
                    best_step = i
                    # save model if achieve a new best valid loss
                    if dist.get_rank() == 0:
                        print('saving model...')
                        save_model(forward_deepnovo,
                                   backward_deepnovo, init_net)

                if dist.get_rank() == 0:
                    writer.add_scalar('validation_loss',
                                      validation_loss, epoch)
                dist.barrier()

                logger.info(f"epoch {epoch} step {i}/{steps_per_epoch}, "
                            f"train perplexity: {perplexity(loss_cpu)}\t"
                            f"validation perplexity: {perplexity(validation_loss)}\tstep time: {step_time}")
            lr_cur = dense_optimizer.state_dict()['param_groups'][0]['lr']

            if dist.get_rank() == 0:

                writer.add_scalar('lr_cur', lr_cur, epoch)
                writer.flush()

            forward_deepnovo.train()
            backward_deepnovo.train()
            start_time = time.time()

        if dist.get_rank() == 0:
            print(f'count={count}')
            writer.close()

    logger.info(f"best model at epoch {best_epoch} step {best_step}")
