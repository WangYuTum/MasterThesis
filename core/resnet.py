'''
    The main structure of the network.

    NOTE: Only train CNN-part of the complete model as a new branch.
'''

from __future__ import division
from __future__ import print_function

import tensorflow as tf

import nn
import sys

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

        if self._data_format is not "NCHW" and self._data_format is not "NHWC":
            sys.exit("Invalid data format. Must be either 'NCHW' or 'NHWC'.")

    def _build_model(self, images, atts, flow_in):
        '''
        :param images: [2,H,W,3], tf.float32, img(t+1), img(t)
        :param atts: [2,H,W,1], tf.int32, att for img(t+1), img(t)
        :param flowin: [1,H,W,2], tf.float32, Flow for img(t)
        :return: segmentation output before softmax
        '''
        # TODO: trainable, test_code
        model = {}

        im_size = tf.shape(images)
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])    # [N,C,H,W]

        ## attention gating on raw images, assuming batch=2
        images = self._att_gate(images, atts)

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 16x
        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # Residual Block B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, images[0:1,:,:,:], 1, 'SAME', shape_dict['B0'])

            # Pooling 1
            model['B0_pooled'] = nn.max_pool2d(self._data_format, model['B0'], 2, 'SAME')

            # Residual Block B1_0, B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'])
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'])

            # Pooling 2
            model['B1_2_pooled'] = nn.max_pool2d(self._data_format, model['B1_2'], 2, 'SAME')

            # Residual Block B2_0, B2_1, B2_2
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1, 1, 128, 256]
            shape_dict['B2']['convs'] = [[3, 3, 128, 256], [3, 3, 256, 256]]
            with tf.variable_scope('B2_0'):
                model['B2_0'] = nn.res_side(self._data_format, model['B1_2_pooled'], shape_dict['B2'])
            for i in range(2):
                with tf.variable_scope('B2_' + str(i + 1)):
                    model['B2_' + str(i + 1)] = nn.res(self._data_format, model['B2_' + str(i)],
                                                       shape_dict['B2']['convs'])

            # Pooling 3
            model['B2_2_pooled'] = nn.max_pool2d(self._data_format, model['B2_2'], 2, 'SAME')

            # Residual Block B3_0 - B3_5
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1, 1, 256, 512]
            shape_dict['B3']['convs'] = [[3, 3, 256, 512], [3, 3, 512, 512]]
            with tf.variable_scope('B3_0'):
                model['B3_0'] = nn.res_side(self._data_format, model['B2_2_pooled'], shape_dict['B3'])
            for i in range(5):
                with tf.variable_scope('B3_' + str(i + 1)):
                    model['B3_' + str(i + 1)] = nn.res(self._data_format, model['B3_' + str(i)],
                                                       shape_dict['B3']['convs'])

            # Pooling 4
            model['B3_5_pooled'] = nn.max_pool2d(self._data_format, model['B3_5'], 2, 'SAME')

            # Residual Block B4_0, B4_1, B4_2
            shape_dict['B4_0'] = {}
            shape_dict['B4_0']['side'] = [1, 1, 512, 1024]
            shape_dict['B4_0']['convs'] = [[3, 3, 512, 512],[3, 3, 512, 1024]]
            with tf.variable_scope('B4_0'):
                model['B4_0'] = nn.res_side(self._data_format, model['B3_5_pooled'], shape_dict['B4_0'])
            shape_dict['B4_23'] = [[3, 3, 1024, 512], [3, 3, 512, 1024]]
            for i in range(2):
                with tf.variable_scope('B4_' + str(i + 1)):
                    model['B4_' + str(i + 1)] = nn.res(self._data_format, model['B4_' + str(i)],
                                                       shape_dict['B4_23'])

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

            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=1)
            else:
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=3)
            with tf.variable_scope('fuse'):
                main_feat_out = nn.conv_layer(self._data_format, concat_seg_feat, 1, 'SAME', [1, 1, 64, 64])
                main_feat_out = nn.bias_layer(self._data_format, main_feat_out, 64)

        with tf.variable_scope('optical_flow'):
            shape_dict['B0_of'] = {}
            shape_dict['B0_of'][0] = [3, 3, 2, 8]
            shape_dict['B0_of'][1] = [1, 1, 8, 8]
            with tf.variable_scope('B0'):
                with tf.variable_scope('conv1'):
                    model['B0_of_1'] = nn.conv_layer(self._data_format, flow_in, 1, 'SAME', shape_dict['B0_of'][0])
                with tf.variable_scope('conv2'):
                    model['B0_of_2'] = nn.conv_layer(self._data_format, model['B0_of_1'], 1, 'SAME',
                                                     shape_dict['B0_of'][1])
            # Pooling 1
            model['B0_of_pooled'] = nn.max_pool2d(self._data_format, model['B0_of_2'], 2, 'SAME')

            shape_dict['B1_of'] = {}
            shape_dict['B1_of'][0] = [3, 3, 8, 16]
            shape_dict['B1_of'][1] = [1, 1, 16, 16]
            with tf.variable_scope('B1'):
                with tf.variable_scope('conv1'):
                    model['B1_of_1'] = nn.conv_layer(self._data_format, model['B0_of_pooled'], 1, 'SAME',
                                                     shape_dict['B1_of'][0])
                with tf.variable_scope('conv2'):
                    model['B1_of_2'] = nn.conv_layer(self._data_format, model['B1_of_1'], 1, 'SAME',
                                                     shape_dict['B1_of'][1])
            # Pooling 2
            model['B1_of_pooled'] = nn.max_pool2d(self._data_format, model['B1_of_2'], 2, 'SAME')

            shape_dict['B2_of'] = {}
            shape_dict['B2_of'][0] = [3, 3, 16, 32]
            shape_dict['B2_of'][1] = [1, 1, 32, 32]
            with tf.variable_scope('B2'):
                with tf.variable_scope('conv1'):
                    model['B2_of_1'] = nn.conv_layer(self._data_format, model['B1_of_pooled'], 1, 'SAME',
                                                     shape_dict['B2_of'][0])
                with tf.variable_scope('conv2'):
                    model['B2_of_2'] = nn.conv_layer(self._data_format, model['B2_of_1'], 1, 'SAME',
                                                     shape_dict['B2_of'][1])
            # Pooling 3
            model['B2_of_pooled'] = nn.max_pool2d(self._data_format, model['B2_of_2'], 2, 'SAME')

            shape_dict['B3_of'] = {}
            shape_dict['B3_of'][0] = [3, 3, 32, 64]
            shape_dict['B3_of'][1] = [1, 1, 64, 64]
            with tf.variable_scope('B3'):
                with tf.variable_scope('conv1'):
                    model['B3_of_1'] = nn.conv_layer(self._data_format, model['B2_of_pooled'], 1, 'SAME',
                                                     shape_dict['B3_of'][0])
                with tf.variable_scope('conv2'):
                    model['B3_of_2'] = nn.conv_layer(self._data_format, model['B3_of_1'], 1, 'SAME',
                                                     shape_dict['B3_of'][1])
            # Pooling 4
            model['B3_of_pooled'] = nn.max_pool2d(self._data_format, model['B3_of_2'], 2, 'SAME')

            shape_dict['B4_of'] = {}
            shape_dict['B4_of'][0] = [3, 3, 64, 128]
            shape_dict['B4_of'][1] = [1, 1, 128, 128]
            with tf.variable_scope('B4'):
                with tf.variable_scope('conv1'):
                    model['B4_of_1'] = nn.conv_layer(self._data_format, model['B3_of_pooled'], 1, 'SAME',
                                                     shape_dict['B4_of'][0])
                with tf.variable_scope('conv2'):
                    model['B4_of_2'] = nn.conv_layer(self._data_format, model['B4_of_1'], 1, 'SAME',
                                                     shape_dict['B4_of'][1])

        with tf.variable_scope('feat_transform'):
            with tf.variable_scope('B0'):
                # concat feat_0 and optical_flow
                if self._data_format == "NCHW":
                    concat_feat_flow_0 = tf.concat([model['B0'], model['B0_of_2']], axis=1)
                else:
                    concat_feat_flow_0 = tf.concat([model['B0'], model['B0_of_2']], axis=3)
                shape_dict['B0_transform'] = [3, 3, 64+8, 64]
                with tf.variable_scope('conv1'):
                    model['B0_trans_1'] = nn.conv_layer(self._data_format, concat_feat_flow_0, 1, 'SAME',
                                                    shape_dict['B0_transform'])
            with tf.variable_scope('B1'):
                if self._data_format == "NCHW":
                    concat_feat_flow_1 = tf.concat([model['B1_2'], model['B1_of_2']], axis=1)
                else:
                    concat_feat_flow_1 = tf.concat([model['B1_2'], model['B1_of_2']], axis=3)
                shape_dict['B1_transform'] = {}
                shape_dict['B1_transform'][0] = [3, 3, 128+16, 128]
                shape_dict['B1_transform'][1] = [3, 3, 128, 128]
                with tf.variable_scope('conv1'):
                    model['B1_trans_1'] = nn.conv_layer(self._data_format, concat_feat_flow_1, 1, 'SAME',
                                                      shape_dict['B1_transform'][0])
                with tf.variable_scope('conv2'):
                    model['B1_trans_2'] = nn.conv_layer(self._data_format, model['B1_trans_1'], 1, 'SAME',
                                                        shape_dict['B1_transform'][1])

            with tf.variable_scope('B2'):
                if self._data_format == "NCHW":
                    concat_feat_flow_2 = tf.concat([model['B2_2'], model['B2_of_2']], axis=1)
                else:
                    concat_feat_flow_2 = tf.concat([model['B2_2'], model['B2_of_2']], axis=3)
                shape_dict['B2_transform'] = {}
                shape_dict['B2_transform'][0] = [3, 3, 256+32, 256]
                shape_dict['B2_transform'][1] = [3, 3, 256, 256]
                shape_dict['B2_transform'][2] = [3, 3, 256, 256]
                with tf.variable_scope('conv1'):
                    model['B2_trans_1'] = nn.conv_layer(self._data_format, concat_feat_flow_2, 1, 'SAME',
                                                        shape_dict['B2_transform'][0])
                with tf.variable_scope('conv2'):
                    model['B2_trans_2'] = nn.conv_layer(self._data_format, model['B2_trans_1'], 1, 'SAME',
                                                        shape_dict['B2_transform'][1])
                with tf.variable_scope('conv3'):
                    model['B2_trans_3'] = nn.conv_layer(self._data_format, model['B2_trans_2'], 1, 'SAME',
                                                        shape_dict['B2_transform'][2])

            with tf.variable_scope('B3'):
                if self._data_format == "NCHW":
                    concat_feat_flow_3 = tf.concat([model['B3_5'], model['B3_of_2']], axis=1)
                else:
                    concat_feat_flow_3 = tf.concat([model['B3_5'], model['B3_of_2']], axis=3)
                shape_dict['B3_transform'] = {}
                shape_dict['B3_transform'][0] = [3, 3, 512+64, 512]
                shape_dict['B3_transform'][1] = [3, 3, 512, 512]
                shape_dict['B3_transform'][2] = [3, 3, 512, 512]
                shape_dict['B3_transform'][3] = [3, 3, 512, 512]
                with tf.variable_scope('conv1'):
                    model['B3_trans_1'] = nn.conv_layer(self._data_format, concat_feat_flow_3, 1, 'SAME',
                                                        shape_dict['B3_transform'][0])
                with tf.variable_scope('conv2'):
                    model['B3_trans_2'] = nn.conv_layer(self._data_format, model['B3_trans_1'], 1, 'SAME',
                                                        shape_dict['B3_transform'][1])
                with tf.variable_scope('conv3'):
                    model['B3_trans_3'] = nn.conv_layer(self._data_format, model['B3_trans_2'], 1, 'SAME',
                                                        shape_dict['B3_transform'][2])
                with tf.variable_scope('conv4'):
                    model['B3_trans_4'] = nn.conv_layer(self._data_format, model['B3_trans_3'], 1, 'SAME',
                                                        shape_dict['B3_transform'][3])

            with tf.variable_scope('B4'):
                if self._data_format == "NCHW":
                    concat_feat_flow_4 = tf.concat([model['B4_2'], model['B4_of_2']], axis=1)
                else:
                    concat_feat_flow_4 = tf.concat([model['B4_2'], model['B4_of_2']], axis=3)
                shape_dict['B4_transform'] = {}
                shape_dict['B4_transform'][0] = [3, 3, 1024+128, 1024]
                shape_dict['B4_transform'][1] = [3, 3, 1024, 512]
                shape_dict['B4_transform'][2] = [3, 3, 512, 512]
                shape_dict['B4_transform'][3] = [3, 3, 512, 1024]
                with tf.variable_scope('conv1'):
                    model['B4_trans_1'] = nn.conv_layer(self._data_format, concat_feat_flow_4, 1, 'SAME',
                                                        shape_dict['B4_transform'][0])
                with tf.variable_scope('conv2'):
                    model['B4_trans_2'] = nn.conv_layer(self._data_format, model['B4_trans_1'], 1, 'SAME',
                                                        shape_dict['B4_transform'][1])
                with tf.variable_scope('conv3'):
                    model['B4_trans_3'] = nn.conv_layer(self._data_format, model['B4_trans_2'], 1, 'SAME',
                                                        shape_dict['B4_transform'][2])
                with tf.variable_scope('conv4'):
                    model['B4_trans_4'] = nn.conv_layer(self._data_format, model['B4_trans_3'], 1, 'SAME',
                                                        shape_dict['B4_transform'][3])


            # aggregate all feature on diff levels
            with tf.variable_scope('B0_trans_side_path'):
                trans_side_0 = nn.conv_layer(self._data_format, model['B0_trans_1'], 1, 'SAME', [3, 3, 64, 16])
                trans_side_0_f = nn.bias_layer(self._data_format, trans_side_0, 16)

            with tf.variable_scope('B1_trans_side_path'):
                trans_side_2 = nn.conv_layer(self._data_format, model['B1_trans_2'], 1, 'SAME', [3, 3, 128, 16])
                trans_side_2 = nn.bias_layer(self._data_format, trans_side_2, 16)
                if self._data_format == "NCHW":
                    trans_side_2 = tf.transpose(trans_side_2, [0, 2, 3, 1])  # To NHWC
                    trans_side_2_f = tf.image.resize_images(trans_side_2, [im_size[1], im_size[2]])  # NHWC
                    trans_side_2_f = tf.transpose(trans_side_2_f, [0, 3, 1, 2])  # To NCHW
                else:
                    trans_side_2_f = tf.image.resize_images(trans_side_2, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B2_trans_side_path'):
                trans_side_4 = nn.conv_layer(self._data_format, model['B2_trans_3'], 1, 'SAME', [3, 3, 256, 16])
                trans_side_4 = nn.bias_layer(self._data_format, trans_side_4, 16)
                if self._data_format == "NCHW":
                    trans_side_4 = tf.transpose(trans_side_4, [0, 2, 3, 1])  # To NHWC
                    trans_side_4_f = tf.image.resize_images(trans_side_4, [im_size[1], im_size[2]])  # NHWC
                    trans_side_4_f = tf.transpose(trans_side_4_f, [0, 3, 1, 2])  # To NCHW
                else:
                    trans_side_4_f = tf.image.resize_images(trans_side_4, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B3_trans_side_path'):
                trans_side_8 = nn.conv_layer(self._data_format, model['B3_trans_4'], 1, 'SAME', [3, 3, 512, 16])
                trans_side_8 = nn.bias_layer(self._data_format, trans_side_8, 16)
                if self._data_format == "NCHW":
                    trans_side_8 = tf.transpose(trans_side_8, [0, 2, 3, 1])  # To NHWC
                    trans_side_8_f = tf.image.resize_images(trans_side_8, [im_size[1], im_size[2]])  # NHWC
                    trans_side_8_f = tf.transpose(trans_side_8_f, [0, 3, 1, 2])  # To NCHW
                else:
                    trans_side_8_f = tf.image.resize_images(trans_side_8, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B4_trans_side_path'):
                trans_side_16 = nn.conv_layer(self._data_format, model['B4_trans_4'], 1, 'SAME', [3, 3, 1024, 16])
                trans_side_16 = nn.bias_layer(self._data_format, trans_side_16, 16)
                if self._data_format == "NCHW":
                    trans_side_16 = tf.transpose(trans_side_16, [0, 2, 3, 1])  # To NHWC
                    trans_side_16_f = tf.image.resize_images(trans_side_16, [im_size[1], im_size[2]])  # NHWC
                    trans_side_16_f = tf.transpose(trans_side_16_f, [0, 3, 1, 2])  # To NCHW
                else:
                    trans_side_16_f = tf.image.resize_images(trans_side_16, [im_size[1], im_size[2]])  # NHWC
            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_trans_feat = tf.concat([trans_side_0_f, trans_side_2_f, trans_side_4_f,
                                               trans_side_8_f, trans_side_16_f], axis=1)
            else:
                concat_trans_feat = tf.concat([trans_side_0_f, trans_side_2_f, trans_side_4_f,
                                               trans_side_8_f, trans_side_16_f], axis=3)
            with tf.variable_scope('fuse'):
                trans_feat_out = nn.conv_layer(self._data_format, concat_trans_feat, 1, 'SAME', [1, 1, 90, 64])
                trans_feat_out = nn.bias_layer(self._data_format, trans_feat_out, 64)

        # classification loss
        with tf.variable_scope('classifier'):
            seg_feat = tf.concat([main_feat_out, trans_feat_out], axis=0) # [2,h,w,64] or [2,64,h,w]
            seg_out = nn.conv_layer(self._data_format, seg_feat, 1, 'SAME', [1, 1, 64, 2])
            seg_out = nn.bias_layer(self._data_format, seg_out, 2)


        return seg_feat, seg_out

    def _att_gate(self, images, atts):
        '''
        :param images: [2,H,W,3] or [2,3,H,W], tf.float32
        :param atts: [2,H,W,1], tf.int32
        :return: gated images
        '''
        if self._data_format == 'NCHW':
            atts = tf.transpose(atts, [0, 3, 1, 2]) # [2,1,H,W]
        att_mask = tf.cast(atts, tf.float32)
        gated_img0 = tf.multiply(images[0:1,:,:,:], att_mask[0:1,:,:,:])
        gated_img1 = tf.multiply(images[1:2,:,:,:], att_mask[1:2,:,:,:])
        gated_imgs = tf.concat([gated_img0, gated_img1], axis=0)
        # gated_img = tf.multiply(images, att_mask)

        return gated_imgs

    def _seg_loss(self, seg_out, seg_gt, seg_weight, att, name):
        '''
        :param seg_out: logits, [4,H,W,2] or [4,2,H,W], tf.float32
        :param seg_gt: [4,H,W,1], tf.int32
        :param seg_weight: [4,H,W,1], tf.float32
        :param name: name for the seg loss
        :return: scalar (balanced cross-entropy), tf.float32
        '''

        loss = self._balanced_cross_entropy(input_tensor=seg_out,
                                            labels=seg_gt,
                                            weight=seg_weight,
                                            att=att)
        tf.summary.scalar('seg_loss_'+name, loss)

        return loss

    def _sim_loss(self, pred, target, mask):
        '''
        :param pred: transformed feat tensor: [1,h,w,64] or [1,64,h,w]
        :param target: target feat tensor: [1,h,w,64] or [1,64,h,w]
        :param mask: mask on target obj(seg_gt of t+1): [1,h,w,1] tf.int32
        :return: scalar loss to be minimized
        '''
        if self._data_format == "NCHW":
            pred = tf.transpose(pred, [0,2,3,1]) # [1,h,w,64]
            target = tf.transpose(target, [0,2,3,1]) # [1,h,w,64]
        # normalize to unit vectors
        pred_vec = tf.nn.l2_normalize(pred, axis=-1, epsilon=1e-12)
        target_vec = tf.nn.l2_normalize(target, axis=-1, epsilon=1e-12)
        # compute cosine dist
        loss = tf.losses.cosine_distance(labels=target_vec, predictions=pred_vec, axis=-1,
                                         weights=tf.cast(mask, tf.float32),
                                         reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
        tf.summary.scalar('sim_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, feed_att, flow_in, mask, weight, global_step, acc_count):
        '''
        :param feed_img: [2,H,W,3], tf.float32: img(t+1), img(t)
        :param feed_seg: [1,H,W,1], tf.int32: img(t+1)
        :param feed_weight: [1,H,W,1], tf.float32: img/seg(t+1)
        :param feed_att: [2,H,W,1], tf.int32: img(t+1), img(t)
        :param flow_in: [1,H,W,2], tf.float32: img(t)
        :param mask: [1,H,W,1], tf.float32, mask for trans_obj and surrounding background, the size should be smaller than feed_att(t+1)
        :param weight: [1,H,W,1], tf.float32, balance weight for trans_obj
        :param global_step: keep track of global train step
        :param acc_count: number of accumulated gradients
        :return: total_loss, train_step_op, grad_acc_op
        '''
        seg_feat, seg_out = self._build_model(feed_img, feed_att, flow_in) # [2,2,H,W] or [2,H,W,2]

        # total_loss:
        #   seg_loss for img(t+1)
        #   seg_loss for img(t)
        #   sim_loss between img(t+1), img(t)
        total_loss = self._seg_loss(seg_out[0:1,:,:,:], feed_seg, feed_weight, feed_att[0:1,:,:,:], 'main') \
                     + self._seg_loss(seg_out[1:2,:,:,:], feed_seg, weight, mask, 'transformed') \
                     + self._sim_loss(seg_feat[1:2,:,:,:], seg_feat[0:1,:,:,:], feed_seg) \
                     + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            seg_pred = seg_out
        tf.summary.image('pred_main', tf.cast(tf.nn.softmax(seg_pred)[0:1, :, :, 1:2], tf.float16))
        tf.summary.image('pred_transformed', tf.cast(tf.nn.softmax(seg_pred)[1:2, :, :, 1:2], tf.float16))

        bp_step, grad_acc_op = self._optimize(total_loss, acc_count, global_step)

        return total_loss, bp_step, grad_acc_op

    def _optimize(self, loss, acc_count, global_step):
        '''
        :param loss: the network loss
        :return: a train op, a grad_acc_op
        '''

        optimizer = tf.train.AdamOptimizer(self._init_lr)
        grads_vars = optimizer.compute_gradients(loss)

        # create grad accumulator for each variable-grad pair
        grad_accumulator = {}
        for idx in range(len(grads_vars)):
            if grads_vars[idx][0] is not None:
                grad_accumulator[idx] = tf.ConditionalAccumulator(grads_vars[idx][0].dtype)
        # apply gradient to each grad accumulator
        layer_lr = nn.param_lr()
        grad_accumulator_op = []
        for var_idx, grad_acc in grad_accumulator.iteritems():
            var_name = str(grads_vars[var_idx][1].name).split(':')[0]
            var_grad = grads_vars[var_idx][0]
            grad_accumulator_op.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        # take average gradients for each variable after accumulating count reaches
        mean_grads_vars = []
        for var_idx, grad_acc in grad_accumulator.iteritems():
            mean_grads_vars.append((grad_acc.take_grad(acc_count), grads_vars[var_idx][1]))

        # apply average gradients to variables
        update_op = optimizer.apply_gradients(mean_grads_vars, global_step=global_step)

        return update_op, grad_accumulator_op

    def _balanced_cross_entropy(self, input_tensor, labels, weight, att):
        '''
        :param input_tensor: the output of final layer, must have shape [batch, C, H, W] or [batch, H, W, C], tf.float32
        :param labels: the gt binary labels, have the shape [batch, H, W, C], tf.int32
        :param weight: shape [batch, H, W, 1], tf.float32
        :return: balanced cross entropy loss, a scalar, tf.float32
        '''

        if self._data_format == "NCHW":
            input_tensor = tf.transpose(input_tensor, [0, 2, 3, 1]) # to NHWC
        input_shape = tf.shape(input_tensor)
        #feed_logits = tf.reshape(input_tensor, [input_shape[0], input_shape[1]*input_shape[2], input_shape[3]])
        #feed_labels = tf.reshape(labels, [input_shape[0], input_shape[1]*input_shape[2]])
        #feed_weight = tf.reshape(weight, [input_shape[0], input_shape[1]*input_shape[2]])

        att_mask = tf.reshape(att, [input_shape[0]*input_shape[1]*input_shape[2]])
        bool_mask = tf.cast(att_mask, tf.bool)
        feed_weight = tf.reshape(weight, [input_shape[0]*input_shape[1]*input_shape[2]])
        feed_weight = tf.boolean_mask(feed_weight, bool_mask)
        feed_labels = tf.reshape(labels, [input_shape[0]*input_shape[1]*input_shape[2]])
        feed_labels = tf.boolean_mask(feed_labels, bool_mask)
        feed_logits = tf.reshape(input_tensor, [input_shape[0]*input_shape[1]*input_shape[2], input_shape[3]])
        feed_logits = tf.boolean_mask(feed_logits, bool_mask)

        cross_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=feed_labels, logits=feed_logits)
        balanced_loss = tf.multiply(cross_loss, feed_weight)

        return tf.reduce_mean(balanced_loss)

    def _l2_loss(self):

        l2_losses = []
        for var in tf.trainable_variables():
            # only regularize conv kernels
            if str(var.name).split(':')[0].find('bias') == -1:
                l2_losses.append(tf.nn.l2_loss(var))
        loss = tf.multiply(self._l2_weight, tf.add_n(l2_losses))
        tf.summary.scalar('l2_loss', loss)

        return loss

    def test(self, images, atts):
        '''
        :param images: batchs/single image have shape [batch, H, W, 3]
        :return: probability map, binary mask
        '''
        net_out = self._build_model(images, atts) # [batch, 2, H, W] or [batch, H, W, 2]
        if self._data_format == "NCHW":
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        prob_map = tf.nn.softmax(net_out) # [batch, H, W, 2]
        pred_mask = tf.argmax(prob_map, axis=3, output_type=tf.int32)  # [batch, H, W]

        return prob_map[:,:,:,1:], pred_mask
