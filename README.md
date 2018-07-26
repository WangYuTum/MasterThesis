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
    
## Trained result (include feat reduce) - No BN, random feed, gate img v2
* Weight init from ResNet-38 ILSVRC-ImageNet
* Data mean/std from Implementation of ResNet-38
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60 * 100 forwards, 60 * 10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* Generate attention area with random size (dilate from 10-35)
* More details, see the code
* Result from 100 ep (maybe try other ep)
  * Same hyper-params as before
  * Use attention gt in testing, but introduced randomized attention area
  * Best Test on val set so far (fine-tune, 500 iters, lr=1e-6)
    * Mean J: 0.84292 (0.86866)
    * Mean F: 0.88229 (0.91311)
  * Result without using attention gt (Failed in tracking probably due to large motion or appearance change)
    * Mean J: 0.58453
    * Mean F: 0.60906
  * TODO:
    Train with random-size attention area and random-shift attention area
  * Conclusion:
    Varied sized attention area is okay; shifted attention area to be verified;
    Still need reasonably accurate attention area.
    
## Trained result (include feat reduce) - No BN, random feed, gate img v3
* Weight init from ResNet-38 ILSVRC-ImageNet
* Data mean/std from Implementation of ResNet-38
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60 * 100 forwards, 60 * 10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* Generate attention area with random size (dilate from 10-35)
* Randomly shift attention area (by -5~+5 pixels in arbitrary direction)
* More details, see the code
* Result from 60 ep (maybe try other ep)
  * Same hyper-params as before
  * Use attention gt in testing, but introduced randomized attention (size, shift)
  * Best Test on val set so far (fine-tune, 500 iters, lr=1e-6)
    * Mean J: 0.85498 (0.84292, 0.86866)
    * Decay J: 0.074299
    * Mean F: 0.89206 (0.88229, 0.91311)

## Trained result (include feat reduce) - No BN, random feed, gate img v4 - flow
* Weight init from ResNet-38 ILSVRC-ImageNet
* Data mean/std from Implementation of ResNet-38
* Gradient accumulate of 10
* lr: 1e-5 (maybe try smaller or larger)
* 100 epochs (1 epoch = 60 * 100 forwards, 60 * 10 backwards)
* resize (0.6-1.0)/flip
* feed out of order, each seq padded to 100, batch=1
* Generate attention area with random size (dilate from 10-35)
* Randomly shift attention area (by -5~+5 pixels in arbitrary direction)
* Randomly add variations to the attention shape
* Randomly add false small attention area (close by cases)
* More details, see the code
* Result from 40 ep (maybe try other ep)
  * Same hyper-params as before
  * Use attention gt in testing, but introduced randomized attention (size, shift, shape-var, false-att)
  * Best Test on val set so far (fine-tune, 500 iters, lr=1e-6)
    * Mean J: 0.8447 (0.85498, 0.856)
    * Decay J: 0.078719 (0.074299, 0.055)
    * Mean F: 0.88514 (0.89206, 0.875)
  * Use optical flow as attention guide (fine-tune, 500 iters, 1000 iters)
    * Mean J: 0.74504
    * Decay J: 0.21986
    * Mean F: 0.76569
    
## Training (include feat reduce) - No BN, random feed, gate img v5
* The CNN-part is the same as v4 (-flow); they are fixed
* Add a small fully connected part to extract object descriptor to refine final segmentation
* The goal is to improve over v4-flow
* Need to train using GT_seg and noisy GT_seg
* Init CNN-part from 40 ep (fixed), train only the added component
* Results:
  * Test on val with randomized gt attention as in v4
  
  * Test on val with flow-attention

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
