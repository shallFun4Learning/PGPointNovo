clean:
	rm -rf build
	rm -f deepnovo_cython_modules.c
	rm -f deepnovo_cython_modules*.so

.PHONY: build
build: clean
	python deepnovo_cython_setup.py build_ext --inplace

.PHONY: train
train:
	CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --train

.PHONY: valid
valid:
	CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --valid

.PHONY: denovo
denovo:
	CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 8 main.py --search_denovo

.PHONY: test
test:
	CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 1 main.py --test

.PHONY: db
db:
	CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" python -m torch.distributed.launch --nproc_per_node 1 main.py --search_db

.PHONY: rec
rec:
	python -m torch.distributed.launch --master_port 8401 --nproc_per_node 4 main.py --train --num_GPUs 4
	 
	