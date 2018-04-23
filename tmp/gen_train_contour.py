# The script generates training set of generic semantic contours on PASCAL Context

import matlab.engine
from scipy.io import loadmat
import sys
import os
import glob
from PIL import Image
from scipy.misc import imsave
import numpy as np
import tensorflow as tf


# define global variables
mat_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_fullscene/trainval/'
rgb_root = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/JPEGImages/'
train_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/train.txt'
val_split = '/usr/stud/wangyu/Downloads/VOC2010/VOC2010_data/VOC2010/ImageSets/Main/val.txt'
max_x = 500
max_y = 500


# start matlab engine
print('Starting Matlab engine ...')
mat_eng = matlab.engine.start_matlab()
print('Matlab engine started.')


# Get train/val split
def get_train_set(train_split):
    with open(train_split) as t:
        arr = t.read().splitlines()
    return arr


# def get_val_set(val_split):
#     with open(val_split) as t:
#         arr = t.read().splitlines()
#     return arr


# Generate binary contour map, given a single .mat path
def get_bin_contour(mat_root, file_name):
    file_path = mat_root + file_name + '.mat'
    bin_contour = mat_eng.get_contour(file_path) # uint8

    return bin_contour


# Get all .mat file list
# def get_mat_list(mat_root):
#     search_path = os.path.join(mat_root, '*.mat')
#     search_files = glob.glob(search_path)
#     search_files.sort()
#
#     return search_files


# Get all train RGB/contour pairs
def get_train_pairs(rgb_root, mat_root, train_split):
    all_pairs = []
    train_list = get_train_set(train_split)
    print('Loading {} images/contours ...'.format(len(train_list)))
    for idx, file_name in enumerate(train_list):
        # get jpg image
        img_obj = Image.open(rgb_root + file_name + '.jpg')
        image = np.array(img_obj, np.uint8)
        # get binary contour
        bin_contour = get_bin_contour(mat_root, file_name)
        bin_contour = np.asarray(bin_contour, np.uint8)
        # as a pair
        pair_data = (image, bin_contour)
        all_pairs.append(pair_data)
        print('loading {}th pair.'.format(idx))
    print('Loading completed.')

    return all_pairs


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def write_single_record(record_writer, data_pair):
    xx = np.array([data_pair[0].shape[0]], np.int32)
    yy = np.array([data_pair[0].shape[1]], np.int32)
    img_paded = np.concatenate((data_pair[0].flatten(), np.zeros(max_x*max_y*3 - xx*yy*3, np.uint8)))
    gt_paded = np.concatenate((data_pair[1].flatten(), np.zeros(max_x*max_y - xx*yy, np.uint8)))
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': _int64_feature(img_paded),
        'gt': _int64_feature(gt_paded),
        'xx': _int64_feature(xx),
        'yy': _int64_feature(yy),
    }))
    record_writer.write(example.SerializeToString())


def generate_tfrecords(pair_list, record_writer):
    for idx, pair_data in enumerate(pair_list):
        write_single_record(record_writer, pair_data)
        print('Write example {}.'.format(idx))
    record_writer.flush()
    record_writer.close()


def main():
    print('Start main script...')
    train_pairs = get_train_pairs(rgb_root, mat_root, train_split)
    mat_eng.quit()
    print('Matlab engine shut down.')
    compression_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    my_writer = tf.python_io.TFRecordWriter('/work/wangyu/PASCAL_Context_train.tfrecord', options=compression_opt)
    generate_tfrecords(train_pairs, my_writer)
    print('Main script done!')


if __name__ == "__main__":
    main()

