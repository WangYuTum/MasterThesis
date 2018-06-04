'''
    The main structure of the network.

    NOTE: Only train CNN-part of the complete model as a new branch.
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
        self._batch = params.get('batch', 1)
        self._l2_weight = params.get('l2_weight', 0.0002)
        self._init_lr = params.get('init_lr', 1e-5)
        # self._base_decay = params.get('base_decay', 1.0)
        # self._sup_decay = params.get('sup_decay', 0.1)
        # self._fuse_decay = params.get('fuse_decay', 0.01)

        if self._data_format is not "NCHW" and self._data_format is not "NHWC":
            sys.exit("Invalid data format. Must be either 'NCHW' or 'NHWC'.")

    def _build_model(self, images, is_train=False):
        '''
        :param images: [1,H,W,3], tf.float32
        :return: segmentation output before softmax
        '''
        model = {}

        im_size = tf.shape(images)
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])    # [N,C,H,W]

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 16x
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

            shape_dict['feat_reduce'] = [1,1,1024,128]
            with tf.variable_scope('feat_reduce'):
                with tf.variable_scope('conv'):
                    model['feat_reduced'] = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME',
                                                        shape_dict['feat_reduce'])
                with tf.variable_scope('bias'):
                    model['feat_reduced'] = nn.bias_layer(self._data_format, model['feat_reduced'], 128)
                model['feat_reduced'] = nn.ReLu_layer(model['feat_reduced']) # [4,h,w,128] or [4,128,h,w]

                # resize to fixed size
                if self._data_format == "NCHW":
                    model['feat_reduced'] = tf.transpose(model['feat_reduced'], [0,2,3,1]) # To NHWC
                    model['feat_resized'] = tf.image.resize_images(model['feat_reduced'], [30, 56])
                    model['feat_resized'] = tf.transpose(model['feat_resized'], [0,3,1,2])  # To NCHW
                else:
                    model['feat_resized'] = tf.image.resize_images(model['feat_reduced'], [30, 56]) # NHWC

            # aggregate all feature on diff levels
            with tf.variable_scope('B1_side_path'):
                side_2 = nn.conv_layer(self._data_format, model['B1_2'], 1, 'SAME', [3, 3, 128, 16])
                side_2 = nn.bias_layer(self._data_format, side_2, 16)
                if self._data_format == "NCHW":
                    side_2 = tf.transpose(side_2, [0,2,3,1]) # To NHWC
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]]) # NHWC
                    side_2_f = tf.transpose(side_2_f, [0,3,1,2]) # To NCHW
                else:
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'], 1, 'SAME', [3, 3, 256, 16])
                side_4 = nn.bias_layer(self._data_format, side_4, 16)
                if self._data_format == "NCHW":
                    side_4 = tf.transpose(side_4, [0, 2, 3, 1])  # To NHWC
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
                    side_4_f = tf.transpose(side_4_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'], 1, 'SAME', [3, 3, 512, 16])
                side_8 = nn.bias_layer(self._data_format, side_8, 16)
                if self._data_format == "NCHW":
                    side_8 = tf.transpose(side_8, [0, 2, 3, 1])  # To NHWC
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
                    side_8_f = tf.transpose(side_8_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 16])
                side_16 = nn.bias_layer(self._data_format, side_16, 16)
                if self._data_format == "NCHW":
                    side_16 = tf.transpose(side_16, [0, 2, 3, 1])  # To NHWC
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]]) # NHWC
                    side_16_f = tf.transpose(side_16_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('resize_side_path'):
                side_reduced = nn.conv_layer(self._data_format, model['feat_resized'], 1, 'SAME', [3, 3, 128, 16])
                side_reduced = nn.bias_layer(self._data_format, side_reduced, 16)
                if self._data_format == "NCHW":
                    side_reduced = tf.transpose(side_reduced, [0, 2, 3, 1])  # To NHWC
                    side_reduced_f = tf.image.resize_images(side_reduced, [im_size[1], im_size[2]]) # NHWC
                    side_reduced_f = tf.transpose(side_reduced_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_reduced_f = tf.image.resize_images(side_reduced, [im_size[1], im_size[2]])  # NHWC

            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f, side_reduced_f], axis=1)
            else:
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f, side_reduced_f], axis=3)
            with tf.variable_scope('fuse'):
                seg_out = nn.conv_layer(self._data_format, concat_seg_feat, 1, 'SAME', [1, 1, 80, 2])
                seg_out = nn.bias_layer(self._data_format, seg_out, 2)

        return seg_out

    def _seg_loss(self, seg_out, seg_gt, seg_weight):
        '''
        :param seg_out: logits, [4,H,W,2] or [4,2,H,W], tf.float32
        :param seg_gt: [4,H,W,1], tf.int32
        :param seg_weight: [4,H,W,1], tf.float32
        :return: scalar (balanced cross-entropy), tf.float32
        '''

        loss = self._balanced_cross_entropy(input_tensor=seg_out,
                                            labels=seg_gt,
                                            weight=seg_weight)
        tf.summary.scalar('seg_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, global_step):
        '''
        :param feed_img: [b,H,W,3], tf.float32
        :param feed_seg: [b,H,W,1], tf.int32
        :param feed_weight: [b,H,W,1], tf.float32
        :param global_step: keep track of global train step
        :return: total_loss, train_step_op
        '''

        seg_out = self._build_model(feed_img, is_train=True) # seg_out shape: [b,H,W,2] or [b,2,H,W] with original input image size
        total_loss = self._seg_loss(seg_out, feed_seg, feed_weight) \
                     + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            seg_pred = seg_out
        tf.summary.image('pred', tf.cast(tf.nn.softmax(seg_pred)[:, :, :, 1:2], tf.float16))

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            bp_step = tf.train.AdamOptimizer(self._init_lr).minimize(total_loss, global_step=global_step)
        print("Model built.")

        return total_loss, bp_step

    def _balanced_cross_entropy(self, input_tensor, labels, weight):
        '''
        :param input_tensor: the output of final layer, must have shape [batch, C, H, W] or [batch, H, W, C], tf.float32
        :param labels: the gt binary labels, have the shape [batch, H, W, C], tf.int32
        :param weight: shape [batch, H, W, 1], tf.float32
        :return: balanced cross entropy loss, a scalar, tf.float32
        '''

        if self._data_format == "NCHW":
            input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1]) # to NHWC
        input_shape = tf.shape(input_tensor)
        feed_logits = tf.reshape(input_tensor, [input_shape[0], input_shape[1]*input_shape[2], input_shape[3]])
        feed_labels = tf.reshape(labels, [input_shape[0], input_shape[1]*input_shape[2]])
        feed_weight = tf.reshape(weight, [input_shape[0], input_shape[1]*input_shape[2]])

        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=feed_labels, logits=feed_logits)
        balanced_loss = tf.multiply(cross_loss, feed_weight)

        return tf.reduce_mean(balanced_loss)

    def _l2_loss(self):

        l2_losses = []
        for var in tf.trainable_variables():
            l2_losses.append(tf.nn.l2_loss(var))
        loss = tf.multiply(self._l2_weight, tf.add_n(l2_losses))
        tf.summary.scalar('l2_loss', loss)

        return loss

    def test(self, images):
        '''
        :param images: batchs/single image have shape [batch, H, W, 3]
        :return: probability map, binary mask
        '''
        net_out = self._build_model(images, is_train=False) # [batch, 2, H, W] or [batch, H, W, 2]
        if self._data_format == "NCHW":
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        prob_map = tf.nn.softmax(net_out) # [batch, H, W, 2]
        pred_mask = tf.argmax(prob_map, axis=3, output_type=tf.int32)  # [batch, H, W]

        return prob_map[:,:,:,1:], pred_mask