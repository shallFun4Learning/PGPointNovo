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
<table border="0" cellpadding="0" cellspacing="0" width="732" style="border-collapse: 
 collapse;table-layout:fixed;width:549pt">
 <colgroup><col width="92" style="mso-width-source:userset;width:69pt">
 <col width="71" style="mso-width-source:userset;width:53.25pt">
 <col width="94" style="mso-width-source:userset;width:70.5pt">
 <col width="99" style="mso-width-source:userset;width:74.25pt">
 <col width="94" span="4" style="mso-width-source:userset;width:70.5pt">
 </colgroup><tbody><tr height="18" style="mso-height-source:userset;height:14.1pt" id="r0">
<td height="16" class="x69" width="92" style="height:12.6pt;width:69pt;">Dataset</td>
<td class="x70" width="71" style="width:53.25pt;">ABRF</td>
<td class="x70" width="94" style="width:70.5pt;">PXD008844</td>
<td class="x70" width="99" style="width:74.25pt;">PXD010559</td>
<td colspan="4" class="x71" width="376" style="border-right:1px solid windowtext;border-bottom:1px solid windowtext;">The merged dataset</td>
 </tr>
 <tr height="18" style="mso-height-source:userset;height:14.1pt" id="r1">
<td height="16" class="x69" style="height:12.6pt;">Data Name</td>
<td class="x70">ABRF</td>
<td class="x70">PXD008844</td>
<td class="x70">PXD010559</td>
<td class="x70">PXD008808</td>
<td class="x70">PXD011246</td>
<td class="x70">PXD012645</td>
<td class="x70">PXD012979</td>
 </tr>
 <tr height="244" style="mso-height-source:userset;height:183.45pt" id="r2">
<td height="242" class="x69" style="height:181.95pt;">Species</td>
<td class="x71">Homo sapiens (Human)</td>
<td class="x71">Mus musculus (Mouse)</td>
<td class="x71">Plasmodium berghei ANKA; Anopheles stephensi (Asian malaria mosquito); Bos taurus (Bovine); Homo sapiens (Human)</td>
<td class="x71">Tursiops truncatus (Atlantic bottle-nosed dolphin) (Delphinus truncatus)</td>
<td class="x71">Homo sapiens (Human)</td>
<td class="x71">&nbsp;Homo sapiens (Human)</td>
<td class="x71">&nbsp;Mus musculus (Mouse)</td>
 </tr>
 <tr height="244" style="mso-height-source:userset;height:183.45pt" id="r3">
<td height="242" class="x72" style="height:181.95pt;">Modifications<br>(Used by PointNovo)</td>
<td class="x71">fixed:C(Carbamidomethylation);<br>variable:M(Oxidation),<br>N(Deamidation),Q(Deamidation)</td>
<td class="x71">fixed:C(Carbamidomethylation);<br>variable:M(Oxidation)</td>
<td class="x71"><font class="font3">fixed:C(Carbamidomethylation);<br>variable:M(Oxidation),N(Deamidation),</font><font class="font22"><br></font><font class="font3">Q(Deamidation),S(Phosphorylation),<br>T(Phosphorylation),Y(Phosphorylation)</font></td>
<td colspan="4" class="x71" style="border-right:1px solid windowtext;border-bottom:1px solid windowtext;">fixed:C(Carbamidomethylation);<br>variable:M(Oxidation)</td>
 </tr>
 <tr height="56" style="mso-height-source:userset;height:42.45pt" id="r4">
<td height="54" class="x72" style="height:40.95pt;">Available Spectrums Count</td>
<td class="x70" x:fmla="=87273+10734+11031">109038</td>
<td class="x70" x:fmla="=100861+12859+11351">125071</td>
<td class="x70" x:fmla="=369243+46021+46497">461761</td>
<td colspan="4" class="x70" x:fmla="=28815+567557" style="border-right:1px solid windowtext;border-bottom:1px solid windowtext;">596372</td>
 </tr>
<!--[if supportMisalignedColumns]-->
 <tr height="0" style="display:none">
  <td width="92" style="width:69pt"></td>
  <td width="71" style="width:53.25pt"></td>
  <td width="94" style="width:70.5pt"></td>
  <td width="99" style="width:74.25pt"></td>
  <td width="94" style="width:70.5pt"></td>
  <td width="94" style="width:70.5pt"></td>
  <td width="94" style="width:70.5pt"></td>
  <td width="94" style="width:70.5pt"></td>
 </tr>
 <!--[endif]-->
</tbody></table>
For links to download the available datasets go to Quick Start.
   
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
  And, download the datasets and files required for the knapsack algorithm. If it is not downloaded, the program will generate it automatically, but this will take some time. For reasons of copyright and fair comparison, we provide links to download the datasets in PointNovo's paper. The datasets covered in the paper and the knapsack files with different modifications can be found [here](https://www.zenodo.org/record/3998873). The test set for PGPointNovo's ablation experiments can be found here(uploading).
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
  Most of the configuration is stored in *config.py* and can be freely altered by the user as appropriate. In general, it is necessary to modify *vocab_reverse* according to the different modifications. In addition, you may also need to specify the knapsack file (refer to **Quick start**).

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

  
