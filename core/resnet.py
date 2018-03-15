'''
    The main structure of the network.

    NOTE: the code is still under developing.
    TODO:
        ** padding in deconv layer: used SAME-osvos. consider VALID suggested from online_tutorial ?
        ** bias layers: added on side path. on backbone net?
        ** load pre-trained ImageNet ckpt. In RGB order
        **** Saver for save/restore weights. Check for the saved weights
        * Learning rate scheduler
        * side supervision loss. read InceptionNet
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

        self._data_format = params.get('data_format', None)
        self._batch = params.get('batch', 2)
        self._l2_weight = params.get('l2_weight', 0.0002)
        self._init_lr = params.get('init_lr', 1e-8)

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
            shape_dict['B3']['convs'] = [[3, 3, 256, 512], [3, 3, 512, 512]]
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
                side_2 = nn.bias_layer(self._data_format, side_2, [16])
                side_2_f = nn.conv_transpose(self._data_format, side_2, 2, 'SAME')
                side_2_f = nn.crop_features(self._data_format, side_2_f, im_size)
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'], 1, 'SAME', [3, 3, 256, 16])
                side_4 = nn.bias_layer(self._data_format, side_4, 16)
                side_4_f = nn.conv_transpose(self._data_format, side_4, 4, 'SAME')
                side_4_f = nn.crop_features(self._data_format, side_4_f, im_size)
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'], 1, 'SAME', [3, 3, 512, 16])
                side_8 = nn.bias_layer(self._data_format, side_8, 16)
                side_8_f = nn.conv_transpose(self._data_format, side_8, 8, 'SAME')
                side_8_f = nn.crop_features(self._data_format, side_8_f, im_size)
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 16])
                side_16 = nn.bias_layer(self._data_format, side_16, 16)
                side_16_f = nn.conv_transpose(self._data_format, side_16, 16, 'SAME')
                side_16_f = nn.crop_features(self._data_format, side_16_f, im_size)

            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_side = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=1)
            else:
                concat_side = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=3)
            with tf.variable_scope('fuse'):
                net_out = nn.conv_layer(self._data_format, concat_side, 1, 'SAME', [1, 1, 64, 2])
                net_out = nn.bias_layer(self._data_format, net_out, 2)

        return net_out

    def train(self, images, gts, weight):
        '''
        :param images: batch of images have shape [batch, H, W, 3] where H, W depend on the scale of dataset
        :param gts: batch of gts have shape [batch, H, W, 1]
        :param weight: batch of balanced weights have shape [batch, H, W, 1]
        :return: a tf.Tensor scalar, a train op
        '''

        net_out = self._build_model(images, True) # [N, C, H, W] or [N, H, W, C]
        total_loss = self._balanced_cross_entropy(net_out, gts, weight) + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # display current predict
        if self._data_format == "NCHW":
            pred_out = tf.transpose(net_out, [0, 2, 3, 1])
        # pred_out = tf.argmax(tf.nn.softmax(pred_out), axis=3) # [batch, H, W]
        # pred_out = tf.expand_dims(pred_out, -1) # [batch, H, W, 1]
        pred_out = pred_out[:,:,:,1:2]
        tf.summary.image('pred', tf.cast(pred_out, tf.float16))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer(self._init_lr).minimize(total_loss)
        print("Model built.")

        return total_loss, train_step

    def test(self, images):
        '''
        :param images: batchs/single image have shape [batch, H, W, 3]
        :return: probability map, binary mask
        '''
        net_out = self._build_model(images, False) # [batch, 2, H, W] or [batch, H, W, 2]
        if self._data_format == "NCHW":
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        prob_map = tf.nn.softmax(net_out) # [batch, H, W, 2]
        pred_mask = tf.argmax(tf.nn.softmax(prob_map), axis=3) # [batch, H, W]

        return prob_map, pred_mask

    def _balanced_cross_entropy(self, input_tensor, labels, weight):
        '''
        :param input_tensor: the output of final layer, must have shape [batch, C, H, W] or [batch, H, W, C], tf.float32
        :param labels: the gt binary labels, have the shape [batch, H, W, C], tf.int32
        :param img_size: python list [H, W]
        :param weight: shape [H, W, 1], tf.float32
        :return: balanced cross entropy loss, a scalar, tf.float32
        '''

        if self._data_format == "NCHW":
            input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1]) # to NHWC
        # Flatten the tensor using convolution
        input_shape = tf.shape(input_tensor)
        feed_logits = tf.reshape(input_tensor, [input_shape[0], input_shape[1]*input_shape[2], input_shape[3]])
        feed_labels = tf.reshape(labels, [input_shape[0], input_shape[1]*input_shape[2]])
        feed_weight = tf.reshape(weight, [input_shape[0], input_shape[1] * input_shape[2]])
        # feed_logits = self._flatten_logits(input_tensor) # [N, H*W, 2]
        # feed_labels = self._flatten_labels(labels, 'tf.int32') # [N, H*W]
        # feed_weight = self._flatten_labels(weight, 'tf.float32') # [N, H*W]

        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=feed_labels, logits=feed_logits)
        balanced_loss = tf.multiply(cross_loss, feed_weight)

        return tf.reduce_mean(balanced_loss)

    def _l2_loss(self):

        l2_losses = []
        for var in tf.trainable_variables():
            l2_losses.append(tf.nn.l2_loss(var))

        return tf.multiply(self._l2_weight, tf.add_n(l2_losses))

    def  _flatten_logits(self, input_tensor):
        '''
        :param input_tensor: Must have shape [N, H, W, 2]
        :return: the same tensor with shape [N, H*W, 2]
        '''
        kernel = tf.Variable([[[[1, 0], [0, 1]]]], dtype=tf.float32, trainable=False, name='flatten_logits_kernel')
        flatten_out = tf.nn.conv2d(input_tensor, kernel, strides=[1, 1, 1, 1], padding='VALID', data_format="NHWC")

        return flatten_out

    def _flatten_labels(self, input_tensor, dtype):
        '''
        :param input_tensor: labels/weights have shape [N, H, W, 1]
        :return: flattened tensor with shape [N, H*W]
        '''
        if dtype == "tf.int32":
            input_tensor = tf.cast(input_tensor, tf.float32)
        kernel = tf.Variable([[[[1]]]], dtype=tf.float32, trainable=False, name='flatten_label_kernel')
        flatten_out = tf.nn.conv2d(input_tensor, kernel, strides=[1,1,1,1], padding='VALID', data_format="NHWC")
        if dtype == "tf.int32":
            flatten_out = tf.cast(flatten_out, tf.int32)

        return flatten_out



































