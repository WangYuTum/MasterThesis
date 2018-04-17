'''
    The data pipeline for DAVIS encapsulated as a class object.

    NOTE: the code is still under developing.

    DONE:
        * Pipeline for training parent network using binary mask
        * Pipeline for fine-tuning/inference parent network using binary mask
    TODO:
        * Pipeline for val/test fine-tune/inference on multi-object mask
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
            mode: 'parent_train_binary'-train network with independent frames
                  'parent_finetune_binary'-finetune network with 1st annotation of the seq
            seq_path: only used in mode 'parent_finetune_binary', e.g. 'DAVIS_root/JPEGImages/480p/bike-packing'
            batch:  1
            tfrecord: None
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._batch = int(params.get('batch', 0))
        self._tfrecord = params.get('tfrecord', None)
        self._seq_path = params.get('seq_path', None)
        self._dataset = None # only used in mode 'parent_train_binary'
        self._scale = 100 # in percentage
        self._map_threads = self._batch * 4
        self._map_buffer = self._batch * 8
        self._seq = None # only used in mode 'parent_finetune_binary'
        self._data_mean = np.array([115.195829334, 114.927476686, 107.725750308]).reshape((1,1,3))
        self._data_std = np.array([64.5572961827, 63.0172054007, 67.0494050908]).reshape((1,1,3))

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode.")
        if self._mode == 'parent_finetune_binary':
            if self._seq_path is None:
                sys.exit("Must specify a seq_path in mode {}".format(self._mode))
        if self._mode == 'parent_train_binary' and self._batch == 0:
            sys.exit("Must specify a batch size in mode {}".format(self._mode))
        if self._mode == 'parent_train_binary':
            if self._tfrecord is None:
                sys.exit("Must specify a path to .tfrecord file in mode {}".format(self._mode))
            else:
                self._scale = int(self._tfrecord.split("_")[2].split(".")[0])

        # Build up the pipeline
        if self._mode == 'parent_train_binary':
            self._dataset = self._build_pipeline()
        elif self._mode == 'parent_finetune_binary':
            self._seq = self._get_seq_frames()
            print('Got {0} frames for seq {1}'.format(len(self._seq), self._seq_path))
        else:
            sys.exit("Not supported mode.")
        if self._mode == 'parent_train_binary' and self._dataset is None:
            sys.exit("Data pipeline not built in mode {}".format(self._mode))
        if self._mode == 'parent_finetune_binary' and self._seq is None:
            sys.exit("Seq {0} data not restored in mode {1}".format(self._seq_path, self._mode))

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
        rgb_img = example['img']
        rgb_img -= self._data_mean
        rgb_img /= self._data_std

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

        # get weight matrix for balanced_cross_entropy loss
        num_pos = tf.reduce_sum(example['gt'])
        num_neg = tf.reduce_sum(1-example['gt'])
        num_total = num_pos + num_neg

        weight_pos = tf.cast(num_neg, tf.float32) / tf.cast(num_total, tf.float32)
        weight_neg = 1.0 - weight_pos

        mat_pos = tf.multiply(tf.cast(example['gt'], tf.float32), weight_pos)
        mat_neg = tf.multiply(tf.cast(1-example['gt'], tf.float32), weight_neg)
        mat_weight = tf.add(mat_pos, mat_neg)

        gt = tf.cast(example['gt'], tf.float32)
        stacked = tf.concat([example['img'], gt, mat_weight], axis=-1) # shape: [H, W, 5]
        stacked = tf.image.random_flip_left_right(stacked)

        # Pack the result
        image = stacked[:,:,0:3]
        gt = tf.cast(stacked[:,:,3:4], tf.int32)
        balanced_mat = stacked[:,:,4:5]
        transformed = {}
        transformed['img'] = image
        transformed['gt'] = gt
        transformed['balanced_mat'] = balanced_mat

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

    ## The following functions are not using .tfrecord.
    ## They are mainly used on seq-by-seq train/fine-tune.
    ## The data are read from disk and stored in main memory for the whole life of the model.

    def _get_seq_frames(self):
        '''
        Example self._seq_path: 'DAVIS_root/JPEGImages/480p/bike-packing'
        :return: A list of numpy image arrays

        NOTE: the returned frames are standardized
        '''
        seq_frames = []
        search_seq_imgs = os.path.join(self._seq_path, "*.jpg")
        files_seq = glob.glob(search_seq_imgs)
        files_seq.sort()
        num_frames = len(files_seq)
        if num_frames == 0:
            sys.exit("Got no frames for seq {}".format(self._seq_path))
        for i in range(num_frames):
            frame = np.array(Image.open(files_seq[i])).astype(np.float32)
            # stardardize
            frame -= self._data_mean
            frame /= self._data_std
            seq_frames.append(frame)

        return seq_frames

    def get_one_shot_pair(self):
        '''
        :return: [img, gt, weight] for the one-shot fine-tuning of the self._seq
        '''
        pair = []
        pair.append(self._seq[0])

        gt_path = os.path.join(self._seq_path.replace('JPEGImages','Annotations'), '00000.png')
        gt = np.array(Image.open(gt_path), dtype=np.int8)
        # convert to binary
        gt_bool = np.greater(gt, 0)
        gt_bin = gt_bool.astype(np.uint8)

        gt = np.expand_dims(gt_bin, axis=-1) # [H,W,1]
        pair.append(gt)

        # Compute balanced weight for training, [H,W,1]
        num_pos = np.sum(gt)
        num_neg = np.sum(1-gt)
        num_total = num_pos + num_neg
        weight_pos = num_neg.astype(np.float32) / num_total.astype(np.float32)
        weight_neg = 1.0 - weight_pos
        mat_pos = np.multiply(gt.astype(np.float32), weight_pos)
        mat_neg = np.multiply((1.0-gt).astype(np.float32), weight_neg)
        mat_weight = np.add(mat_pos, mat_neg)
        pair.append(mat_weight)

        return pair

    def get_test_frames(self):

        frame_list = self._seq[1:]

        return frame_list



