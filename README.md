# PGPointNovo
PGPointNovo, PointNovo with more advanced optimization strategy and supporting data parallelization.

## Usage

  1. I guess that you have be familiar with pointnovo, right? 
  
     1.1 If not, you should first learn about pointnovo or deepnovo which are excellent de novo tools.(https://github.com/volpato30/DeepNovoV2)
  
  2. If you already have pointnovo experience, you can adjust some additional parameters in config. Here are some parameter descriptions.<br>
     `
     use_ranger = True #Using the Ranger optimizer` <br>
     
     
     `
     use_sync = True #Using SyncBatchNormalization`  <br>
  
     
     `
     factor = 0.8 #For the learning rate plan, the steps are reduced in the case of multiple GPUs,
     and 0.8 is a good parameter obtained in the case of 8 gpus. You could try more.
     `
     Please don't forget the parameters that should have been changed when using pointnovo, such as file path, regenerating related files, etc. They 
     
  3. Ok, if you don't care about the deep learning technology, after configuring the relevant parameters, you just need to enter` make rec `on the terminal to start running. Richer parameters for data parallelization are set in the makefile file.
