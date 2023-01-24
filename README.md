# PGPointNovo
## About
PGPointNovo, PointNovo with more advanced optimization strategy and supporting data parallelization. Extensive experiments conducted on multiple datasets of different sizes demonstrate that PGPointNovo achieves profound speedups against the excellent approach without sacrificing precision, recall and generalizability.

## Hardware Support
We ran PGPointNovo on a single node of the super-computer in Naional Center for Protein Sciences (Beijing). This node has two 2.6GHz Intel Xeon processors, eight Tesla V100 16GB GPUs and 256 GB RAM and runs under CentOS 7.6.

## Installation
1. Use the git command or download the zip file locally.
    ~~~
    git clone  https://github.com/shallFun4Learning/PGPointNovo.git
    ~~~
2. Dependency

   *(If you can run PointNovo normally, you can skip this step.)*
   
   Base Dependency:
    >    Python 3.7.11
    >    
    >    PyTorch 1.7.1
    >
    >    cudnn7.6.5

   We recommend using conda for environment management.
   
   a. To create a new environment.
   ~~~
   conda create -n YOUR_ENV_NAME python=3.7
   ~~~
   
   b. Switch to the created environment.
   ~~~
   conda activate YOUR_ENV_NAME
   ~~~
   
   c. Install PyTorch
   ~~~
   conda install pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cudatoolkit=10.2 -c pytorch
   ~~~
   or see the [PyTorch website](https://pytorch.org/get-started/previous-versions/).
   
   d. About other dependencies
   
   Now, let's install the PointNovo dependencies.
   ~~~
   pip install bio pyteomics
   ~~~
   ~~~
   conda install cython pandas tensorboard dataclasses
   ~~~

## DataSets

   
## Usage
  Ranger is a non-invasive optimiser. We have made a high level wrapper for PointNovo, so the usage is essentially the same as PointNovo, which eliminates the need for additional learning costs for the user. PointNovo's README could be found [here](https://github.com/shallFun4Learning/PGPointNovo/blob/main/README4PointNovo.md).
 ### Quick start  
 Before starting, you should run the following commands to create the necessary files.
  ~~~
  make clean
  ~~~
  ~~~
  make build
  ~~~
  And, Download the datasets and files required for the knapsack algorithm. If it is not downloaded, the program will generate it automatically, but this will take some time.For reasons of copyright and fair comparison, we provide links to download the datasets in PointNovo's paper.The datasets covered in the paper and the knapsack files with different modifications can be found [here](https://www.zenodo.org/record/3998873). The test set for PGPointNovo's ablation experiments can be found [here]().
  To train model：
  ~~~
  make train
  ~~~
  To *de novo* :
  The inference stage is here.
  ~~~
  make denovo
  ~~~
  To get performance reslut:
  ~~~
  make test
  ~~~
  
  ### Advanced Use Tutorials
  #### About PointNovo
  Most of the configuration is stored in *config.py* and can be freely altered by the user as appropriate. In general, it is necessary to modify *vocab_reverse* according to the different modifications.In addition, you may also need to specify the knapsack file (refer to **Quick start**).

  In PointNovo, the dataset file path needs to be specified manually,such as: 
  > input_spectrum_file_train = "PXD008844/spectrums.mgf"
  > 
  > input_feature_file_train = "PXD008844/features.csv.identified.train.nodup"
  > 
  > input_spectrum_file_valid = "PXD008844/spectrums.mgf"
  > 
  > input_feature_file_valid = "PXD008844/features.csv.identified.valid.nodup"
  > 
  > denovo_input_spectrum_file = "PXD008844/spectrums.mgf"
  > 
  > denovo_input_feature_file = "PXD008844/features.csv.identified.test.nodup"

  
  #### About PGPointNovo
  PGPointNovo inherits the way PointNovo is configured, but with some additional encapsulation.
  Our recommended form of document organisation is as follows.
  ```
  PGPointNovo
  ├── PXD008844
  │   ├── features.csv.identified.test.nodup
  │   ├── features.csv.identified.train.nodup
  │   ├── features.csv.identified.valid.nodup
  │   ├── spectrums.mgf
  │   └── spectrums.mgf.location.pytorch.pkl
  ├── runs
  ├── train
  │   ├── backward_deepnovo.pth
  │   └── forward_deepnovo.pth
  └── ...Other files
```
  With this dataset file naming rule, you only need to specify the dataset name in the *config.py* and the dataset file path will be generated automatically. Alternatively, you can use the manual specification of the file path, just like with PointNovo.
  
  In addition to this, there are two important interfaces in the config file.
  
  *use_ranger* is used to control whether the Ranger optimiser is used instead of the Adam optimiser. As an option, we recommend using it when there is a loss of precision/recall.
  
   *use_sync* is used to control whether to use the SyncBatchNorm instead of the BatchNorm in PointNovo. In the case of distributed data parallel training on multi GPUs, the BatchNorm calculation process (statistical mean and variance) is independent between processes, i.e. each process can only see the local GlobalBatchSize/NumGpu size data. We usually recommend that this is left set to True.
   
## Supports
Feel free to submit an issue or contact the author (sfun@foxmail.com) if you encounter any problems during use.
Happy Year of the Rabbit :-)

  
