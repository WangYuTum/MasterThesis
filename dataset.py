'''
    The data pipeline for DAVIS encapsulated as a class object.

    NOTE: The code/branch is only used for training a generic contour capturing network.
    The resulting inference/weights may be used for DAVIS specific tasks, such as: full-scene contour prediction,
    primary-obj-contour capturing, etc.

    DONE:
        * Pipeline for training parent network using binary contour
        * Pipeline for inference generic contours on PASCAL Context
    TODO(under developing):
        * Pipeline for inference generic contours on DAVIS dataset
        * Get/sort image indices for inference DAVIS contours

'''

# NOTE: The .tfrecord files are located on /work/wangyu/ on hpccremers4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from PIL import Image
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
            mode: 'train_contour'-train network with independent frames
                  'inf_PASCAL_val_contour'-inference on PASCAL Context val dataset
                  'inf_DAVIS_contour'-inference on DAVIS train/val/test datasets
            batch:  1
            tfrecord: None
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._batch = int(params.get('batch', 0))
        self._tfrecord = params.get('tfrecord', None)
        self._dataset = None
        self._map_threads = self._batch * 4
        self._map_buffer = self._batch * 8
        # the mean/std are computed over PASCAL Context trainval sets
        self._data_mean = np.array([116.737144764, 111.800790818, 103.241458398]).reshape((1,1,3))
        self._data_std = np.array([69.8958826812, 69.1125806592, 72.5705461484]).reshape((1,1,3))

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode.")
        if self._batch == 0:
            sys.exit("Must specify a batch size.")
        if self._tfrecord is None:
            sys.exit("Must specify a path to .tfrecord file.")

        # Build up the pipeline
        if self._mode == 'train_contour' or \
                self._mode == 'inf_PASCAL_val_contour' or \
                self._mode == 'inf_DAVIS_contour':
            self._dataset = self._build_pipeline()
        else:
            sys.exit("Not supported mode.")
        if self._dataset is None:
            sys.exit("Data pipeline build failed.")

    def _parse_single_record(self, record):
        '''
        :param record: tf.string, serilized
        :return: data_dict
        '''

        # TFrecords format for DAVIS binary annotation dataset
        if self._mode == 'train_contour' or self._mode == 'inf_PASCAL_val_contour':
            record_features = {
                "img": tf.FixedLenFeature([500*500, ], tf.int64),
                "gt": tf.FixedLenFeature([500*500, ], tf.int64),
                "xx": tf.FixedLenFeature([1, ], tf.int64),
                "yy": tf.FixedLenFeature([1, ], tf.int64),
            }
        elif self._mode == 'inf_DAVIS_contour':
            # TODO
            pass
        else:
            sys.exit("Current mode not supported.")

        data_dict = {}
        out_data = tf.parse_single_example(serialized=record, features=record_features)

        if self._mode == 'train_contour' or self._mode == 'inf_PASCAL_val_contour':
            xx = np.reshape(out_data['xx'], ())
            yy = np.reshape(out_data['yy'], ())
            size_true = xx * yy
            img_flat = out_data['img'][:size_true]
            gt_flat = out_data['gt'][:size_true]
            data_dict['img'] = tf.cast(np.reshape(img_flat, (xx, yy)), tf.float32)
            data_dict['gt'] = tf.cast(np.reshape(gt_flat, (xx, yy)), tf.int32)
            data_dict['xx'] = tf.cast(xx, tf.int32)
            data_dict['yy'] = tf.cast(yy, tf.int32)
        elif self._mode == 'inf_DAVIS_contour':
            # TODO
            pass

        return data_dict

    def _image_std(self, example):
        '''
        :param example: parsed record - a dict
        :return: a dict where RGB image is standardized
        '''
        rgb_img = example['img']
        rgb_img -= self._data_mean
        rgb_img /= self._data_std

        # Pack the result
        tranformed = {}
        tranformed['img'] = rgb_img
        tranformed['gt'] = example['gt']
        if self._mode == 'train_contour' or self._mode == 'inf_PASCAL_val_contour':
            tranformed['xx'] = example['xx']
            tranformed['yy'] = example['yy']
        elif self._mode == 'inf_DAVIS_contour':
            # TODO
            pass

        return tranformed

    def _train_contour_transform(self, example):
        '''
        :param example: a standardized dict
        :return: a dict where rgb/gt is transformed

        Transformations:
            - Randomly flip img/gt together
            - Randomly resize img/gt together, ratio: [0.7, 1.3]
        '''

        # get weight matrix for balanced_cross_entropy loss
        num_pos = tf.reduce_sum(example['gt'])
        num_neg = tf.reduce_sum(1 - example['gt'])
        num_total = num_pos + num_neg
        weight_pos = tf.cast(num_neg, tf.float32) / tf.cast(num_total, tf.float32)
        weight_neg = 1.0 - weight_pos
        mat_pos = tf.multiply(tf.cast(example['gt'], tf.float32), weight_pos)
        mat_neg = tf.multiply(tf.cast(1 - example['gt'], tf.float32), weight_neg)
        mat_weight = tf.add(mat_pos, mat_neg)

        # stack and randomly flip
        gt = tf.cast(example['gt'], tf.float32)
        stacked = tf.concat([example['img'], gt, mat_weight], axis=-1)  # shape: [H, W, 5]
        stacked = tf.image.random_flip_left_right(stacked)

        # unstack and randomly resize
        image = stacked[:,:,0:3] # tf.float32
        gt = tf.cast(stacked[:, :, 3:4], tf.int32) # tf.int32
        balanced_mat = stacked[:, :, 4:5] # tf.float32
        [ratio] = np.random.randint(7, 13, 1)
        new_xx = tf.cast(example['xx']*ratio/10, tf.int32)
        new_yy = tf.cast(example['yy']*ratio/10, tf.int32)
        new_size = [new_xx, new_yy]
        image = tf.image.resize_images(image, new_size, tf.image.ResizeMethod.BILINEAR)
        gt = tf.image.resize_images(gt, new_size, tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        balanced_mat = tf.image.resize_images(balanced_mat, new_size, tf.image.ResizeMethod.BILINEAR)

        # pack result
        transformed = {}
        transformed['img'] = image
        transformed['gt'] = gt
        transformed['balanced_mat'] = balanced_mat

        return transformed

    def _build_train_contour_pipeline(self, tfrecord_file):
        '''
        :param tfrecord_file:
        :return: a tf.contrib.dataset object

        NOTE: The .tfrecord is compressed as "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(tfrecord_file, "GZIP")
        dataset = dataset.repeat()
        dataset = dataset.map(self._parse_single_record, num_threads=self._map_threads,
                              output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._image_std, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._train_contour_transform, num_threads=self._map_threads,
                              output_buffer_size=self._map_buffer)
        dataset = dataset.shuffle(buffer_size=1500)
        dataset = dataset.batch(self._batch)

        return dataset

    def _build_inf_pascal_pipeline(self, tfrecord_file):
        '''
        :param tfrecord_file:
        :return: a tf.contrib.dataset object

        NOTE: The .tfrecord is compressed as "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(tfrecord_file, "GZIP")
        dataset = dataset.map(self._parse_single_record, num_threads=self._map_threads,
                              output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._image_std, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.batch(self._batch)

        return dataset

    def _build_inf_davis_pipeline(self, tfrecord_file):
        '''
        :param tfrecord_file:
        :return: a tf.contrib.dataset object

        NOTE: The .tfrecord is compressed as "GZIP"
        '''
        dataset = tf.contrib.data.TFRecordDataset(tfrecord_file, "GZIP")
        dataset = dataset.map(self._parse_single_record, num_threads=self._map_threads,
                              output_buffer_size=self._map_buffer)
        dataset = dataset.map(self._image_std, num_threads=self._map_threads, output_buffer_size=self._map_buffer)
        dataset = dataset.batch(self._batch)

        return dataset

    def _build_pipeline(self):

        if self._mode == 'train_contour':
            dataset = self._build_train_contour_pipeline(tfrecord_file=self._tfrecord)
            print("train_contour pipeline built. Loaded tfrecord: {}".format(self._tfrecord))
        elif self._mode == 'inf_PASCAL_val_contour':
            dataset = self._build_inf_pascal_pipeline(tfrecord_file=self._tfrecord)
            print("inf_PASCAL_val_contour pipeline built. Loaded tfrecord: {}".format(self._tfrecord))
        elif self._mode == 'inf_DAVIS_contour':
            dataset = self._build_inf_davis_pipeline(tfrecord_file=self._tfrecord)
            print("inf_DAVIS_contour pipeline built. Loaded tfrecord: {}".format(self._tfrecord))
        else:
            sys.exit('Mode not supported.')

        return dataset

    def next_batch(self):
        '''
        :return: a dict:
            * next_batch['img'] = [batch_size, H, W, 3], tf.float32
            * next_batch['gt'] = [batch_size, H, W, 1], tf.int32, binary
        '''
        batch_iterator = self._dataset.make_one_shot_iterator()
        next_batch = batch_iterator.get_next()

        return next_batch


