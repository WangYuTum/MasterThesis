# This is a test script on generate PASCAL Context image/contour pairs
# The matlab engine cannot work in Jupyter.

import matlab.engine
from scipy.io import loadmat
import sys
import os
import glob
from PIL import Image
from scipy.misc import imsave
import numpy as np
import datetime


# define global variables
mat_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_fullscene/trainval/'
rgb_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/JPEGImages/'
train_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/train.txt'
val_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/val.txt'


# start matlab engine
print('Starting Matlab engine ...')
ts = datetime.datetime.now()
mat_eng = matlab.engine.start_matlab()
tf = datetime.datetime.now()
te = tf - ts
print('Matlab engine started, {}'.format(te))


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
def get_bin_contour(mat_root, file_name):
    file_path = mat_root + file_name + '.mat'
    bin_contour = mat_eng.get_contour(file_path) # uint8

    return bin_contour

# Get all .mat file list
def get_mat_list(mat_root):
    search_path = os.path.join(mat_root, '*.mat')
    search_files = glob.glob(search_path)
    search_files.sort()

    return search_files

# Get all train RGB/contour pairs
def get_train_pairs(rgb_root, mat_root, train_split):
    all_pairs = []
    train_list = get_train_set(train_split)
    max_x = 0
    max_y = 0
    for file_name in train_list:
        # get jpg image
        img_obj = Image.open(rgb_root + file_name + '.jpg')
        image = np.array(img_obj, np.uint8)
        xx = image.shape[0]
        yy = image.shape[1]
        if xx > max_x:
            max_x = xx
        if yy > max_y:
            max_y = yy
        # get binary contour
        bin_contour = get_bin_contour(mat_root, file_name)
        bin_contour = np.asarray(bin_contour, np.uint8)
        # as a pair
        pair_data = (image, bin_contour)
        all_pairs.append(pair_data)
    print('max_x: {}'.format(max_x))
    print('max_y: {}'.format(max_y))

    return all_pairs

# For testing
def main():
    print('Start main ...')

    mat_file_list = get_mat_list(mat_root)
    print('Number of mat files: {}.'.format(len(mat_file_list)))
    train_list = get_train_set(train_split)
    print('Numer of train split: {}'.format(len(train_list)))
    val_list = get_val_set(val_split)
    print('Numer of train split: {}'.format(len(val_list)))

    print('Loading images/contour_gt ...')
    train_pairs = get_train_pairs(rgb_root, mat_root, train_split)
    print('Loaded images/contour_gt.')
    print('type of image: {}'.format(type(train_pairs[0][0])))
    print('type of image: {}'.format(type(train_pairs[0][1])))


if __name__ == "__main__":
    main()

