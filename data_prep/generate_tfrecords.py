'''
This file generates 3 tfrecords with each having different image scales: [0.5, 0.8, 1.0]
The names of the tfrecords: [davis_train_50, davis_train_80, davis_train_100]
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import sys
import os
from scipy.misc import imresize
from scipy.ndimage.morphology import binary_dilation
import multiprocessing

train_root = '/usr/stud/wangyu/DAVIS17_train_val'
train_list_txt = '../Notebook/train_list.txt'
train_gt_txt = '../Notebook/train_gt_list.txt'
NUM_PROCESSES = 3


def get_file_list(train_root, train_list_txt, train_gt_txt):
    '''
        Get all RGB images and corresponding labels
    '''
    with open(train_list_txt) as t:
        train_frames = t.read().splitlines()
    with open(train_gt_txt) as t:
        gt_frames = t.read().splitlines()
    if len(train_frames) != len(gt_frames):
        sys.exit("Train/Gt length do not match!")
    else:
        print("Got {0} train/gt frames.".format(len(train_frames)))

    len_train = len(train_frames)
    train_pair_list = []
    for i in range(len_train):
        file_img = os.path.join(train_root, train_frames[i])
        file_gt = os.path.join(train_root, gt_frames[i])
        train_pair_list.append([file_img, file_gt])

    return train_pair_list


def load_img(file_pair, scale):
    ''' Input:  a list of length 2: [img_name, gt_name]
        Return: 3 arrays: [img_arr, gt_arr, boundary_arr]
            img_arr: [480,854,3]
            gt_arr: [480,854]
            gt_boundary: [480,854]
        Note: resize to train/gt images to [480,854] first, then rescale
    '''
    img = Image.open(file_pair[0])
    img_sc = imresize(img, (480, 854))
    image = np.array(img_sc, dtype=np.uint8)

    gt = Image.open(file_pair[1])
    gt_label = np.array(gt, dtype=np.uint8)
    gt_label_bool = np.greater(gt_label, 0)
    gt_label_bin = gt_label_bool.astype(np.uint8)
    gt_label_bin_sc = imresize(gt_label_bin, (480, 854), interp='nearest')
    gt_label_bin = np.array(gt_label_bin_sc, dtype=np.uint8)
    # compute binary pixel-wise boundary, width=4
    gt_label_grad = np.gradient(gt_label_bin)
    gt_label_grad = np.transpose(gt_label_grad, [1, 2, 0])
    img_boundary = np.greater(gt_label_grad[:, :, 1] + gt_label_grad[:, :, 0], 0).astype(np.uint8)
    img_boundary = binary_dilation(img_boundary)

    img_sc = imresize(image, scale)
    gt_sc = imresize(gt_label_bin, scale, interp='nearest')
    img_boundary_sc = imresize(img_boundary, scale, interp='nearest')
    new_img = np.array(img_sc, dtype=np.uint8)
    new_gt = np.array(gt_sc, dtype=np.uint8)
    new_boundary = np.array(img_boundary_sc, dtype=np.uint8)

    return new_img, new_gt, new_boundary


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_a_data_dict(file_pair, scale):
    ''' Input: [img_name, gt_name]
        Return: Dict
    '''
    data_dict = {}
    data_dict['img'], data_dict['gt'], data_dict['boundary'] = load_img(file_pair, scale)

    return data_dict


def write_single_record(record_writer, data_dict):
    ''' Input: record_writer, single data_dict
        Return: No return.
    '''
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': _int64_feature(data_dict['img'].flatten()),
        'gt': _int64_feature(data_dict['gt'].flatten()),
        'boundary': _int64_feature(data_dict['boundary'].flatten()), }))
    record_writer.write(example.SerializeToString())


def generate_tfrecords(files_list, record_writer, scale):
    for example_file_list in files_list:
        # Get all necessary data
        data_dict = wrap_a_data_dict(example_file_list, scale)
        # Write a single record/exampel to .tfrecord
        write_single_record(record_writer, data_dict)
        example_name = example_file_list[0].replace('.jpg', '')
        print('Write example: {}'.format(example_name))
    record_writer.flush()
    record_writer.close()

def main():
    train_file_list = get_file_list(train_root, train_list_txt, train_gt_txt)
    num_processes = NUM_PROCESSES

    compression_option = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.GZIP)
    train_writer_100 = tf.python_io.TFRecordWriter('/work/wangyu/davis_train_100.tfrecord', options=compression_option)
    train_writer_80 = tf.python_io.TFRecordWriter('/work/wangyu/davis_train_80.tfrecord', options=compression_option)
    train_writer_50 = tf.python_io.TFRecordWriter('/work/wangyu/davis_train_50.tfrecord', options=compression_option)

    process_pool = []
    process_pool.append(multiprocessing.Process(target=generate_tfrecords, args=(train_file_list, train_writer_100, 1.0)))
    process_pool.append(multiprocessing.Process(target=generate_tfrecords, args=(train_file_list, train_writer_80, 0.8)))
    process_pool.append(multiprocessing.Process(target=generate_tfrecords, args=(train_file_list, train_writer_50, 0.5)))

    for i in range(num_processes):
        process_pool[i].start()
    for i in range(num_processes):
        process_pool[i].join()

    print("Generate train tfrecords done!")

if __name__ == "__main__":
    main()
