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
  
## Trained result (include feat reduce) - No BN, random feed
* Weight init from ResNet-38 ILSVRC-ImageNet
* Data mean/std from Implementation of ResNet-38
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60 * 100 forwards, 60 * 10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* More details, see the code
* Result from 100 ep (maybe try other ep)
  * Same hyper-params as before
  * Best Test on val set so far (fine-tune, 500 iters, lr=1e-6)
    * Mean J: 0.70597
    * Mean F: 0.71415
    
## Trained result (include feat reduce) - No BN, random feed, gate img
* Weight init from ResNet-38 ILSVRC-ImageNet
* Data mean/std from Implementation of ResNet-38
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60 * 100 forwards, 60 * 10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* More details, see the code
* Result from 100 ep (maybe try other ep)
  * Same hyper-params as before
  * Best Test on val set so far (fine-tune, 500 iters, lr=1e-6)
    * Mean J: 0.86866
    * Mean F: 0.91311
  * ISSUE: Very sensitive to the size of the attention area. Re-train
    in another new branch where size of the attention area is randomized.

## Re-training use previous configurations (fixed attention size), visualize activitations.
 

## TODO
* Feed seq by seq without BN (major)
* Side supervision (depends)

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
