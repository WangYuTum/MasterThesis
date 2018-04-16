# This is a test script on generate PASCAL Context image/contour pairs
# The matlab engine cannot work in Jupyter.

import matlab.engine
from scipy.io import loadmat
import sys
import os
import glob

# define global variables
mat_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_fullscene/trainval/'
rgb_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/JPEGImages'
train_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/train.txt'
val_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/val.txt'


# start matlab engine
mat_eng = matlab.engine.start_matlab()


# Get train/val split
def get_train_set(train_split):
    with open(train_split) as t:
        arr = t.read().splitlines()
    return arr


def get_val_set(val_split):
    with open(val_split) as t:
        arr = t.read().splitlines()
    return arr


# Generate binary contour map, given a single .mat path
def get_bin_contour(file_name):
    file_path = mat_root + file_name + '.mat'
    bin_contour = mat_eng.get_contour(file_path) # uint8


# Get all train RGB/contour pairs
def get_train_pairs():
    train_list = get_train_set()
    for item in train_list:
