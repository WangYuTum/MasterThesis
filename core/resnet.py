'''
    The main structure of the network.

    NOTE: the code is still under developing.
    TODO:
        ** padding in deconv layer ??? SAME-osvos, VALID-online_tutorial
        ** bias layers ??? must add on side path. on backbone net?
        ** side supervision loss. read InceptionNet
        * load pre-trained ImageNet ckpt. Pay attention to RGB means.
        * Saver for save/restore weights. Check for each layer whether first created or restored.
        * Learning rate scheduler
        * Apply diff lr for diff layers
        * Estimator ...
'''

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

import nn
import sys
import os

class ResNet():
    def __init__(self, params):
        '''
        :param params(dict) keys:
            feed_ckpt: string, e.g. '../data/ckpts/imgnet.ckpt'
            data_format: either "NCHW" or "NHWC"
        '''

        self._feed_ckpt = params.get('feed_ckpt', None)
        self._data_format = params.get('data_format', None)

        if self._feed_ckpt is None:
            sys.exit("Invalid feed .ckpt")
        if self._data_format is not "NCHW" and self._data_format is not "NHWC":
            sys.exit("Invalid data format. Must be either 'NCHW' or 'NHWC'.")

    def _build_model(self, images, is_train=False):
        '''
        :param image: image in RGB format
        :param is_train: either True or False
        :return: a model dict containing all Tensors
        '''
        model = {}
        if is_train:
            dropout=True
        else:
            dropout=False
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])

        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # Residual Block B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, images, 1, 'SAME', shape_dict['B0'])

            # Pooling 1
            model['B0_pooled'] = nn.max_pool2d(self._data_format, model['B0'], 2, 'SAME')

            # Residual Block B1_0, B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'], is_train)
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'], is_train)

            # Pooling 2
            model['B1_2_pooled'] = nn.max_pool2d(self._data_format, model['B1_2'], 2, 'SAME')

            # Residual Block B2_0, B2_1, B2_2
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1, 1, 128, 256]
            shape_dict['B2']['convs'] = [[3, 3, 128, 256], [3, 3, 256, 256]]
            with tf.variable_scope('B2_0'):
                model['B2_0'] = nn.res_side(self._data_format, model['B1_2_pooled'], shape_dict['B2'], is_train)
            for i in range(2):
                with tf.variable_scope('B2_' + str(i + 1)):
                    model['B2_' + str(i + 1)] = nn.res(self._data_format, model['B2_' + str(i)],
                                                       shape_dict['B2']['convs'], is_train)

            # Pooling 3
            model['B2_2_pooled'] = nn.max_pool2d(self._data_format, model['B2_2'], 2, 'SAME')

            # Residual Block B3_0 - B3_5
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1, 1, 256, 512]
            shape_dict['B3']['convs'] = [[3, 3, 512, 512], [3, 3, 512, 512]]
            with tf.variable_scope('B3_0'):
                model['B3_0'] = nn.res_side(self._data_format, model['B2_2_pooled'], shape_dict['B3'], is_train)
            for i in range(5):
                with tf.variable_scope('B3_' + str(i + 1)):
                    model['B3_' + str(i + 1)] = nn.res(self._data_format, model['B3_' + str(i)],
                                                       shape_dict['B3']['convs'], is_train)

            # Pooling 4
            model['B3_5_pooled'] = nn.max_pool2d(self._data_format, model['B3_5'], 2, 'SAME')

            # Residual Block B4_0, B4_1, B4_2
            shape_dict['B4_0'] = {}
            shape_dict['B4_0']['side'] = [1, 1, 512, 1024]
            shape_dict['B4_0']['convs'] = [[3, 3, 512, 512],[3, 3, 512, 1024]]
            with tf.variable_scope('B4_0'):
                model['B4_0'] = nn.res_side(self._data_format, model['B3_5_pooled'], shape_dict['B4_0'], is_train)
            shape_dict['B4_23'] = [[3, 3, 1024, 512], [3, 3, 512, 1024]]
            for i in range(2):
                with tf.variable_scope('B4_' + str(i + 1)):
                    model['B4_' + str(i + 1)] = nn.res(self._data_format, model['B4_' + str(i)],
                                                       shape_dict['B4_23'], is_train)

            # add side conv path and upsample, crop to image size
            im_size = tf.shape(images)
            with tf.variable_scope('B1_side_path'):
                side_2 = nn.conv_layer(self._data_format, model['B1_2'], 1, 'SAME', [3, 3, 128, 16])
                side_2_f = nn.conv_transpose(self._data_format, side_2, 2, 'SAME')
                side_2_f = nn.crop_features(self._data_format, side_2_f, im_size)
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'], 1, 'SAME', [3, 3, 256, 16])
                side_4_f = nn.conv_transpose(self._data_format, side_4, 4, 'SAME')
                side_4_f = nn.crop_features(self._data_format, side_4_f, im_size)
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'], 1, 'SAME', [3, 3, 512, 16])
                side_8_f = nn.conv_transpose(self._data_format, side_8, 8, 'SAME')
                side_8_f = nn.crop_features(self._data_format, side_8_f, im_size)
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 16])
                side_16_f = nn.conv_transpose(self._data_format, side_16, 16, 'SAME')
                side_16_f = nn.crop_features(self._data_format, side_16_f, im_size)

            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_side = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=1)
            else:
                concat_side = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=3)
            with tf.variable_scope('fuse'):
                net_out = nn.conv_layer(self._data_format, concat_side, 1, 'SAME', [1, 1, 64, 1])

        return net_out

    def train(self, images, gts, params):


        total_loss = None

        return total_loss
























