# MasterThesis

## Attention on single object
The branch test attention mechanism on single object.

## ISSUES
* RNN resolution might be too low
* Loss unstable
 * Loss might be sensitive to object size
 * Attention sparsity (remove or modify)
* Resize features at the end of CNN might make accurate mask localisation difficult

## TODO
* Fine-tune during inference
 

## Build Enviroment (no sudo privilege)
* Ubuntu 16.04
* Python 2.7.12
  * numpy 1.14.3
  * pip 10.0.1
  * wheel 0.31.1
  * python2.7-dev
* GCC 5.4.0
* Bazel 0.10.0
* Tensorflow r1.8
* GPU
  * CUDA 9.1
  * CUDNN 7.1.3
  * TITAN XP
    * Compute Capability: 6.1
