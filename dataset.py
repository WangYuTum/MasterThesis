'''
    The data pipeline for DAVIS encapsulated as a class object.

    NOTE: the code is still under developing.
    TODO:
        * Pipeline for val/test fine-tune/inference
'''

# NOTE: The .tfrecord files are located on /work/wangyu/ on hpccremers4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import sys
import os
import glob
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imread

class DAVIS_dataset():
    def __init__(self, params):
        '''
        :param params(dict) keys:
            mode: 'parent_train_binary'
            batch:  2
            tfrecord: None
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._batch = int(params.get('batch', 0))
        self._tfrecord = params.get('tfrecord', None)
        self._dataset = None
        self._scale = 100 # in percentage
        self._map_threads = self._batch * 2
        self._map_buffer = self._batch * 4

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode.")
        if self._batch == 0:
            sys.exit("Must specify a batch size.")
        if self._tfrecord is None:
            sys.exit("No valid .tfrecord file nor sequence list.")
        else:
            self._scale = int(self._tfrecord.split("_")[2])

        # Build up the pipeline
        if self._mode == 'parent_train_binary':
            self._dataset = self._build_pipeline()
        if self._dataset is None:
            sys.exit("Data pipeline not built.")

    def _parse_single_record(self, record):
        '''
        :param record: tf.string, serilized
        :return: data_dict
        '''

        # TFrecords format for DAVIS binary annotation dataset
        if self._mode == 'parent_train_binary':
            img_H = int(480 * self._scale / 100)
            img_W = int(854 * self._scale / 100)
            record_features = {
                "img": tf.FixedLenFeature([img_H,img_W,3], tf.int64),
                "gt": tf.FixedLenFeature([img_H,img_W,1], tf.int64)
            }
        else:
            sys.exit("Current mode not supported.")

        data_dict = {}
        out_data = tf.parse_single_example(serialized=record, features=record_features)

        # Cast RGB pixels to tf.float32, gt to tf.int32
        data_dict['img'] = tf.cast(out_data['img'], tf.float32)
        data_dict['gt'] = tf.cast(out_data['gt'], tf.int32)

        return data_dict

    def _image_std(self, example):
        '''
        :param example: parsed record - a dict
        :return: a dict where RGB image is standardized
        '''

        mean = np.array([115.195829334, 114.927476686, 107.725750308]).reshape((1,1,3))
        std = np.array([64.5572961827, 63.0172054007, 67.0494050908]).reshape((1,1,3))

        rgb_img = example['img']
        rgb_img -= mean
        rgb_img /= std

        # Pack the result
        tranformed = {}
        tranformed['img'] = rgb_img
        tranformed['gt'] = example['gt']

        return tranformed

    def _parent_binary_train_transform(self, example):
        '''
        :param example: a standardized dict
        :return: a dict where rgb/gt is transformed

        Transformations:
            - Randomly flip img/gt together
        '''

        gt = tf.cast(example['gt'], tf.float32)
        stacked = tf.concat([example['img'], gt], axis=-1) # shape: [H, W, 4]
        stacked = tf.image.random_flip_left_right(stacked)

        # Pack the result
        image = stacked[:,:,0:3]
        gt = tf.cast(stacked[:,:,3:4], tf.int32)
        transformed = {}
        transformed['img'] = image
        transformed['gt'] = gt

        return  transformed

    def _build_parent_binary_pipeline(self, tfrecord_file):
        '''
        :param tfrecord_file: .tfrecord file path, comes from self._tfrecord
        :return: a tf.contrib.dataset object

        NOTE: The .tfrecord is compressed as "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(tfrecord_file, "GZIP")
        dataset = dataset.repeat()
        dataset = dataset.map(self._parse_single_record, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._image_std, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._parent_binary_train_transform, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.shuffle(buffer_size=1500)
        dataset = dataset.batch(self._batch)

        return  dataset

    def _build_pipeline(self):

        if self._mode == 'parent_train_binary':
            dataset = self._build_parent_binary_pipeline(tfrecord_file=self._tfrecord)
            print("Train_parent_binary pipeline built. Load tfrecord: {}".format(self._tfrecord))
        else:
            sys.exit("Not supported mode.")

        return dataset

    def get_iterator(self):

        iter = self._dataset.make_one_shot_iterator()

        return iter

    def next_batch(self):

        batch_iterator = self._dataset.make_one_shot_iterator()
        next_batch = batch_iterator.get_next()

        return next_batch




