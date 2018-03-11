'''
    The main structure of the network.

    NOTE: the code is still under developing.
    TODO:
        * load pre-trained ImageNet ckpt. Pay attention to RGB means.
        * Saver for save weights
        * Estimator ...
        * Apply diff lr for diff layers
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

        # Residual Block B0 (x1)
        with tf.variable_scope('main'):
            with tf.variable_scope('B0'):
                model['B0'] =







