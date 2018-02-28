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
import glob
from scipy.misc import imresize
import multiprocessing

train_root = '/usr/stud/wangyu/DAVIS17_train_val'
NUM_PROCESSES = 3


def get_file_list(train_root):
    '''
        Get all RGB images and corresponding labels
    '''
    search_train_img = os.path.join(train_root, "JPEGImages", "480p", "*", "*.jpg")
    search_train_gt = os.path.join(train_root, "Annotations", "480p", "*", "*.png")

    files_train_img = glob.glob(search_train_img)
    files_train_gt = glob.glob(search_train_gt)

    files_train_img.sort()
    files_train_gt.sort()

    train_img_len = len(files_train_img)
    train_img_gt = len(files_train_gt)

    if (train_img_len != train_img_gt):
        sys.exit('Length of train/val files do not match!')
    else:
        print('Got {0} train/gt files.'.format(train_img_len))

    # Group train/gt pairs
    train_file_list = []
    for i in range(train_img_len):
        train_file_list.append([files_train_img[i], files_train_gt[i]])

    return train_file_list


def load_img(file_pair, scale):
    ''' Input:  a list of length 2: [img_name, gt_name]
        Return: two arrays: [img_arr, gt_arr]
            img_arr: [480,910,3]
            gt_arr: [480,910]
    '''
    img = Image.open(file_pair[0])
    image = np.array(img, dtype=np.uint8)

    gt = Image.open(file_pair[1])
    gt_label = np.array(gt, dtype=np.uint8)
    gt_label_bool = np.greater(gt_label, 0)
    gt_label_bin = gt_label_bool.astype(np.uint8)

    img_sc = imresize(image, scale)
    gt_sc = imresize(gt_label_bin, scale)
    new_img = np.array(img_sc, dtype=np.uint8)
    new_gt = np.array(gt_sc, dtype=np.uint8)

    return new_img, new_gt


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def wrap_a_data_dict(file_pair, scale):
    ''' Input: [img_name, gt_name]
        Return: Dict
    '''
    data_dict = {}
    data_dict['img'], data_dict['gt'] = load_img(file_pair, scale)

    return data_dict


def write_single_record(record_writer, data_dict):
    ''' Input: record_writer, single data_dict
        Return: No return.
    '''
    example = tf.train.Example(features=tf.train.Features(feature={
        'img': _int64_feature(data_dict['img'].flatten()),
        'gt': _int64_feature(data_dict['gt'].flatten()), }))
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
    train_file_list = get_file_list(train_root)
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
