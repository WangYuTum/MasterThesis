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
max_xx = 500
max_yy = 500


# start matlab engine
print('Starting Matlab engine ...')
mat_eng = matlab.engine.start_matlab()
print('Matlab engine started.')


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
def get_train_pairs(rgb_root, mat_root, train_split, val_split):
    all_pairs = []
    train_list = get_train_set(train_split)
    val_list = get_val_set(val_split)
    trainval_list = train_list + val_list
    for file_name in trainval_list:
        # get jpg image
        img_obj = Image.open(rgb_root + file_name + '.jpg')
        image = np.array(img_obj, np.uint8)
        # get binary contour
        # bin_contour = get_bin_contour(mat_root, file_name)
        # bin_contour = np.asarray(bin_contour, np.uint8)
        bin_contour = np.zeros(1, np.uint8) # use this dumpy contour when computing r,g,b means of the dataset
        # as a pair
        pair_data = (image, bin_contour)
        all_pairs.append(pair_data)

    return all_pairs


def compute_rgb_mean(pairs):

    num_img = len(pairs)
    mean_r = 0.0
    mean_g = 0.0
    mean_b = 0.0

    sum_r_all = []
    sum_g_all = []
    sum_b_all = []

    num_pixels_all_single_channel = 0
    print('Computing means ...')
    for i in range(num_img):
        image = pairs[i][0]
        xx = image.shape[0]
        yy = image.shape[1]
        sum_r_all.append(np.sum(image[:,:,0]))
        sum_g_all.append(np.sum(image[:,:,1]))
        sum_b_all.append(np.sum(image[:,:,2]))
        num_pixels_all_single_channel += xx * yy
    sum_r = np.sum(sum_r_all)
    sum_g = np.sum(sum_g_all)
    sum_b = np.sum(sum_b_all)
    mean_r = sum_r / num_pixels_all_single_channel
    mean_g = sum_g / num_pixels_all_single_channel
    mean_b = sum_b / num_pixels_all_single_channel

    print('Num pixels per channel: {}'.format(num_pixels_all_single_channel))
    print('Sum in R: {}'.format(sum_r))
    print('Sum in G: {}'.format(sum_g))
    print('Sum in B: {}'.format(sum_b))
    print('R: {0}, G: {1}, B: {2}'.format(mean_r, mean_g, mean_b))

    return mean_r, mean_g, mean_b


def compute_rgb_std(pairs, mean_r, mean_g, mean_b):
    num_img = len(pairs)
    std_r = 0.0
    std_g = 0.0
    std_b = 0.0

    std_r_all = []
    std_g_all = []
    std_b_all = []
    num_pixels_all_single_channel = 0
    print('Computing stds ...')
    for i in range(num_img):
        image = pairs[i][0]
        xx = image.shape[0]
        yy = image.shape[1]
        std_r_all.append(np.sum(np.square(image[:,:,0] - mean_r)))
        std_g_all.append(np.sum(np.square(image[:,:,1] - mean_g)))
        std_b_all.append(np.sum(np.square(image[:,:,2] - mean_b)))
        num_pixels_all_single_channel += xx * yy
    std_r = np.sqrt(np.sum(std_r_all) / num_pixels_all_single_channel)
    std_g = np.sqrt(np.sum(std_g_all) / num_pixels_all_single_channel)
    std_b = np.sqrt(np.sum(std_b_all) / num_pixels_all_single_channel)

    print('std_R:{0}, std_G:{1}, std_B:{2}'.format(std_r, std_g, std_b))

    return std_r, std_g, std_b


# For testing
def main():
    print('Start main ...')

    print('Loading images/contour_gt ...')
    train_pairs = get_train_pairs(rgb_root, mat_root, train_split, val_split)
    mat_eng.quit()
    print('Loaded images/contour_gt. In total: {}. Matlab engine shut down.'.format(len(train_pairs)))
    mean_r, mean_g, mean_b = compute_rgb_mean(train_pairs)
    std_r, std_g, std_b = compute_rgb_std(train_pairs, mean_r, mean_g, mean_b)


if __name__ == "__main__":
    main()

