'''
    The main structure of the network.

    NOTE: the code is still under developing.
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
        self._l1_att = params.get('l1_att', 0.0001)
        self._init_lr = params.get('init_lr', 1e-5)
        self._base_decay = params.get('base_decay', 1.0)
        self._sup_decay = params.get('sup_decay', 0.1)
        self._fuse_decay = params.get('fuse_decay', 0.01)

        if self._data_format is not "NCHW" and self._data_format is not "NHWC":
            sys.exit("Invalid data format. Must be either 'NCHW' or 'NHWC'.")

    def _build_model(self, images, atts, att_oracle):
        '''
        :param images: [4,H,W,3], tf.float32, f0, f1, f2, f3
        :param atts: [2,H,W,1], tf.int32, f0, f2
        :param att_oracle: [2,H,W,1], tf.int32, f1, f3
        :return: attention and segmentation branch output before softmax
        '''
        model = {}

        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])
        im_size = tf.shape(images)

        ## attention gating on raw images
        images = self._att_gate(images, atts, att_oracle)


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
                model['feat_resized'] = tf.image.resize_images(model['feat_reduced'], [30, 56])

            f_0 = model['feat_resized'][0:1,:,:,:] # [1,30,56,128] or [1,128,30,56]
            f_1 = model['feat_resized'][1:2,:,:,:] # [1,30,56,128] or [1,128,30,56]
            f_2 = model['feat_resized'][2:3,:,:,:]
            f_3 = model['feat_resized'][3:4,:,:,:]

        with tf.variable_scope('segmentation'):
            # go to conv2dLSTM
            f_0_2 = tf.concat([f_0, f_2], axis=0)
            with tf.variable_scope('seg_lstm2d'):
                out_lstm2d_seg = nn.lstm_conv2d(self._data_format, f_0_2) # [2,30,56,128] or [2,128,30,56]
            shape_dict['lstm2d_decode'] = [1,1,128,128]
            with tf.variable_scope('lstm2d_decode'):
                model['lstm2d_decode'] = nn.conv_layer(self._data_format, out_lstm2d_seg, 1,
                                                       'SAME', shape_dict['lstm2d_decode'])

            # aggregate all feature on diff levels
            # CNN part has batch of 4, while lstm has batch of 2,
            # extract f0, f2 from CNN part for segmentation branch backprop
            with tf.variable_scope('B1_side_path'):
                B1_f02 = tf.concat([model['B1_2'][0:1,:,:,:], model['B1_2'][2:3,:,:,:]],axis=0)
                side_2 = nn.conv_layer(self._data_format, B1_f02, 1, 'SAME', [3, 3, 128, 16])
                side_2 = nn.bias_layer(self._data_format, side_2, 16)
                side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]])
                # side_2_f = nn.conv_transpose(self._data_format, side_2, [16, 16], 2, 'SAME')
                # side_2_f = nn.crop_features(self._data_format, side_2_f, im_size)
            with tf.variable_scope('B2_side_path'):
                B2_f02 = tf.concat([model['B2_2'][0:1, :, :, :], model['B2_2'][2:3, :, :, :]], axis=0)
                side_4 = nn.conv_layer(self._data_format, B2_f02, 1, 'SAME', [3, 3, 256, 16])
                side_4 = nn.bias_layer(self._data_format, side_4, 16)
                side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]])
                # side_4_f = nn.conv_transpose(self._data_format, side_4, [16, 16], 4, 'SAME')
                # side_4_f = nn.crop_features(self._data_format, side_4_f, im_size)
            with tf.variable_scope('B3_side_path'):
                B3_f02 = tf.concat([model['B3_5'][0:1, :, :, :], model['B3_5'][2:3, :, :, :]], axis=0)
                side_8 = nn.conv_layer(self._data_format, B3_f02, 1, 'SAME', [3, 3, 512, 16])
                side_8 = nn.bias_layer(self._data_format, side_8, 16)
                side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]])
                # side_8_f = nn.conv_transpose(self._data_format, side_8, [16, 16], 8, 'SAME')
                # side_8_f = nn.crop_features(self._data_format, side_8_f, im_size)
            with tf.variable_scope('B4_side_path'):
                B4_f02 = tf.concat([model['B4_2'][0:1, :, :, :], model['B4_2'][2:3, :, :, :]], axis=0)
                side_16 = nn.conv_layer(self._data_format, B4_f02, 1, 'SAME', [3, 3, 1024, 16])
                side_16 = nn.bias_layer(self._data_format, side_16, 16)
                side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]])
                # side_16_f = nn.conv_transpose(self._data_format, side_16, [16, 16], 16, 'SAME')
                # side_16_f = nn.crop_features(self._data_format, side_16_f, im_size)
            with tf.variable_scope('lstm_decoded'):
                side_lstm = nn.conv_layer(self._data_format, model['lstm2d_decode'], 1, 'SAME', [3, 3, 128, 16])
                side_lstm = nn.bias_layer(self._data_format, side_lstm, 16)
                side_lstm_f = tf.image.resize_images(side_lstm, [im_size[1], im_size[2]])
                # side_lstm_f = nn.conv_transpose(self._data_format, side_lstm, [16, 16], 16, 'SAME')
                # side_lstm_f = nn.crop_features(self._data_format, side_lstm_f, im_size)

            # concat and linearly fuse
            if self._data_format == "NCHW":
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f, side_lstm_f], axis=1)
            else:
                concat_seg_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f, side_lstm_f], axis=3)
            with tf.variable_scope('fuse'):
                seg_out = nn.conv_layer(self._data_format, concat_seg_feat, 1, 'SAME', [1, 1, 80, 2])
                seg_out = nn.bias_layer(self._data_format, seg_out, 2)


        with tf.variable_scope('attention'):
            # fuse visual features from f0 and f1
            if self._data_format == "NCHW":
                dual_feat0 = tf.concat([model['lstm2d_decode'][0:1,:,:,:], f_1], axis=1) # [1,256,30,56]
                dual_feat1 = tf.concat([model['lstm2d_decode'][1:2,:,:,:], f_3], axis=1)  # [1,256,30,56]
            else:
                dual_feat0 = tf.concat([model['lstm2d_decode'][0:1,:,:,:], f_1], axis=3)  # [1,256,30,56]
                dual_feat1 = tf.concat([model['lstm2d_decode'][1:2,:,:,:], f_3], axis=3)  # [1,256,30,56]
            # fuse dual feat and reduce
            dual_feats = tf.concat([dual_feat0, dual_feat1], axis=0) # [2,256,56,30] or [2,30,56,256]
            with tf.variable_scope('fuse'):
                dual_feats = nn.conv_layer(self._data_format, dual_feats, 1, 'SAME', [1,1,256,256])
                dual_feats = nn.bias_layer(self._data_format, dual_feats, 256)
                dual_feats = nn.ReLu_layer(dual_feats)
            with tf.variable_scope('reduce'):
                dual_feats = nn.conv_layer(self._data_format, dual_feats, 1, 'SAME', [1,1,256,128])
                dual_feats = nn.bias_layer(self._data_format, dual_feats, 128)
                dual_feats = nn.ReLu_layer(dual_feats)
            with tf.variable_scope('att_lstm'):
                out_lstm2d_att = nn.lstm_conv2d(self._data_format, dual_feats) # [2,56,30,128] or [2,128,56,30]
            with tf.variable_scope('lstm2d_decode'):
                att_lstm_decode = nn.conv_layer(self._data_format, out_lstm2d_att, 1,
                                                'SAME', [1,1,128,128])
            with tf.variable_scope('up'):
                att_out = nn.conv_layer(self._data_format, att_lstm_decode, 1, 'SAME', [1,1,128,2])
                att_out = nn.bias_layer(self._data_format, att_out, 2)
                att_out = tf.image.resize_images(att_out, [im_size[1], im_size[2]])
                # att_out = nn.conv_transpose(self._data_format, att_out, [2, 2], 16, 'SAME')
                # att_out = nn.crop_features(self._data_format, att_out, im_size)

        return att_out, seg_out

    def _att_gate(self, images, atts, att_oracle):
        '''
        :param images: [4,H,W,3] or [4,3,H,W], tf.float32
        :param atts: [2,H,W,1], tf.int32
        :param att_oracle: [2,H,W,1], tf.int32
        :return: stacked gated f0, f1, f2, f3: [4,H,W,3] or [4,3,H,W]
        '''

        if self._data_format == 'NCHW':
            atts = tf.transpose(atts, [0, 3, 1, 2])
            att_oracle = tf.transpose(att_oracle, [0, 3, 1, 2])
        att0_mask = tf.cast(atts[0:1,:,:,:], tf.float32)
        att2_mask = tf.cast(atts[1:2,:,:,:], tf.float32)
        att_oracle_mask1 = tf.cast(att_oracle[0:1,:,:,:], tf.float32)
        att_oracle_mask3 = tf.cast(att_oracle[1:2,:,:,:], tf.float32)
        # Gate
        gated_f0 = tf.multiply(images[0:1,:,:,:], att0_mask)
        gated_f1 = tf.multiply(images[1:2,:,:,:], att_oracle_mask1)
        gated_f2 = tf.multiply(images[2:3,:,:,:], att2_mask)
        gated_f3 = tf.multiply(images[3:4,:,:,:], att_oracle_mask3)
        stacked = tf.concat([gated_f0, gated_f1, gated_f2, gated_f3], axis=0)

        return stacked

    def _att_loss(self, att_out, att_gt, att_weight):
        '''
        :param att_out: logits, [2,H,W,2] or [2,2,H,W], tf.float32
        :param att_gt: [2,H,W,1], tf.int32
        :param att_weight: [2,H,W,1], tf.float32
        :return: scalar (balanced cross-entropy), tf.float32
        '''

        loss = self._balanced_cross_entropy(input_tensor=att_out,
                                            labels=att_gt,
                                            weight=att_weight) * 0.1
        tf.summary.scalar('att_loss', loss)

        return loss

    def _att_sparsity(self, att_out):
        '''
        :param att_out: logits, [2,H,W,2] or [2,2,H,W], tf.float32
        :return: l1 norm on the softmax of positive prob, tf.float32
        '''
        if self._data_format == "NCHW":
            att_out = tf.transpose(att_out, [0,2,3,1])
        att_prob = tf.nn.softmax(att_out)[:,:,:,1:2] # [2,h,w,1]
        loss1 = tf.norm(att_prob[0:1,:,:,:], ord=1) * self._l1_att
        loss3 = tf.norm(att_prob[1:2,:,:,:], ord=1) * self._l1_att
        loss = 0.0001  * (loss1 + loss3) / 2.0
        tf.summary.scalar('att_sparsity', loss)

        return loss

    def _att_coverage(self, att_out, seg_gt):
        '''
        :param att_out: logits, [2,H,W,2] or [2,2,H,W], tf.float32
        :param seg_gt: [2,H,W,1], tf.int32
        :return: scalar, coverage over seg_gt, tf.float32

        NOTE: Consider seq of length 2 for now.
        '''

        if self._data_format == "NCHW":
            att_out = tf.transpose(att_out, [0,2,3,1]) # To NHWC
        seg_gt = tf.cast(seg_gt, tf.float32)
        att_prob = tf.nn.softmax(att_out)
        att_prob1 = att_prob[0:1,:,:,1:2] # [1,h,w,1]
        att_prob1 = tf.multiply(att_prob1, seg_gt[0:1,:,:,:])
        att_prob1 = tf.minimum(att_prob1, 0.7) # probability >= 0.7 will not contrib to loss
        # coverage ratio cannot exceed 1.5
        ratio_cover1 = tf.minimum(tf.reduce_sum(att_prob1)/tf.reduce_sum(seg_gt[0:1,:,:,:]), 1.5)

        att_prob2 = att_prob[1:2,:,:,1:2] # [1,h,w,1]
        att_prob2 = tf.multiply(att_prob2, seg_gt[1:2,:,:,:])
        att_prob2 = tf.minimum(att_prob2, 0.7)
        ratio_cover2 = tf.minimum(tf.reduce_sum(att_prob2) / tf.reduce_sum(seg_gt[1:2,:,:,:]), 1.5)

        loss = -(tf.log(ratio_cover1) + tf.log(ratio_cover2)) / 2.0
        tf.summary.scalar('att_cover', loss)

        return loss

    def _seg_loss(self, seg_out, seg_gt, seg_weight):
        '''
        :param seg_out: logits, [2,H,W,2] or [2,2,H,W], tf.float32
        :param seg_gt: [2,H,W,1], tf.int32
        :param seg_weight: [2,H,W,1], tf.float32
        :return: scalar (balanced cross-entropy), tf.float32

        NOTE: the balanced weights are computed within the attention area of f0
              area beyond the attention is set to 0s
        '''

        loss = self._balanced_cross_entropy(input_tensor=seg_out,
                                            labels=seg_gt,
                                            weight=seg_weight)
        tf.summary.scalar('seg_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, feed_att, feed_att_oracle, global_step):
        '''
        :param feed_img: [4,H,W,3], tf.float32; f0, f1, f2, f3
        :param feed_seg: [4,H,W,1], tf.int32; s0, s1, s2, s3
        :param feed_weight: [4,H,W,1], tf.float32; w_s0, w_att1, w_s2, w_att3
        :param feed_att: [4,H,W,1], tf.int32; a0, a1, a2, a3
        :param feed_att_oracle: [2,H,W,1], tf.int32; a01, a23
        :param global_step: keep track of global train step
        :return: total_loss, train_step_op
        '''

        att_02 = tf.concat([feed_att[0:1,:,:,:], feed_att[2:3,:,:,:]], axis=0) # [2,h,w,1]
        att_out, seg_out = self._build_model(feed_img, att_02, feed_att_oracle)
        # att_out, seg_out both shape: [2,H,W,2] or [2,2,H,W] with original input image size
        att_13 = tf.concat([feed_att[1:2,:,:,:], feed_att[3:4,:,:,:]], axis=0) # [2,h,w,1]
        weight_13 = tf.concat([feed_weight[1:2,:,:,:], feed_weight[3:4,:,:,:]], axis=0) # [2,h,w,1]
        feed_seg02 = tf.concat([feed_seg[0:1,:,:,:], feed_seg[2:3,:,:,:]], axis=0)
        feed_seg13 = tf.concat([feed_seg[1:2,:,:,:], feed_seg[3:4,:,:,:]], axis=0)
        weight02 = tf.concat([feed_weight[0:1,:,:,:] ,feed_weight[2:3,:,:,:]], axis=0)
        total_loss = self._att_loss(att_out, att_13, weight_13) \
                    + self._att_sparsity(att_out) \
                    + self._seg_loss(seg_out, feed_seg02, weight02) \
                    + self._l2_loss() \
                    + self._att_coverage(att_out, feed_seg13)
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            att_pred = tf.transpose(att_out, [0,2,3,1])
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            att_pred = att_out
            seg_pred = seg_out
        tf.summary.image('att_pred1', tf.cast(tf.nn.softmax(att_pred[0:1,:,:,:])[:,:,:,1:2], tf.float16))
        tf.summary.image('att_pred3', tf.cast(tf.nn.softmax(att_pred[1:2,:,:,:])[:,:,:,1:2], tf.float16))
        tf.summary.image('seg_pred0', tf.cast(tf.nn.softmax(seg_pred[0:1,:,:,:])[:,:,:,1:2], tf.float16))
        tf.summary.image('seg_pred2', tf.cast(tf.nn.softmax(seg_pred[1:2,:,:,:])[:,:,:,1:2], tf.float16))

        # train_step_op, grad_acc_op = self._optimize(total_loss, acc_count, global_step)
        train_step_op = tf.train.AdamOptimizer(self._init_lr).minimize(total_loss, global_step=global_step)

        return total_loss, train_step_op

    def test(self, images):
        '''
        :param images: batchs/single image have shape [batch, H, W, 3]
        :return: probability map, binary mask
        '''
        net_out, sup_out = self._build_model(images, False) # [batch, 2, H, W] or [batch, H, W, 2]
        if self._data_format == "NCHW":
            net_out = tf.transpose(net_out, [0, 2, 3, 1])
        prob_map = tf.nn.softmax(net_out) # [batch, H, W, 2]
        pred_mask = tf.argmax(prob_map, axis=3, output_type=tf.int32)  # [batch, H, W]

        return prob_map[:,:,:,1:], pred_mask

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

        return tf.multiply(self._l2_weight, tf.add_n(l2_losses))

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