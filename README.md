# MasterThesis

## Attention on single object
A sub-branch of Attention-bin.
Train full-sized images (batch=4) seq by seq only using CNN part (including feature reduce/resize).
* Test on val set (without fine-tune)
  * Mean J: 0.175
  * Mean F: 0.139
* Test on val set(with fine-tune, 1000 iters, 1e-4)
  * Mean J: 0.449
  * Mean F: 0.385
  
## Train ongoing (include feat reduce)
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60*100 forwards, 60*10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* More details, see the code
* Test on val set (fine-tune, 500/1000 iters, lr=?)
  * Mean J:
  * Mean F: 
 

## TODO
* Try BN (depends)
* Side supervision (depends)
* Different feed pattern (later)

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
