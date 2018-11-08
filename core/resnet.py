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

    def _build_model(self, images, atts, flow_in, train_flag):
        '''
        :param images: [2,H,W,3], tf.float32, img(t+1), img(t)
        :param atts: [2,H,W,1], tf.int32, att for img(t+1), img(t)
        :param flowin: [1,H,W,2], tf.float32, Flow for img(t)
        :param train_flag: 0:train main, 1:train trans, 2:train all, 3:no train all
        :return: activation_map output before softmax
        '''
        model = {}

        im_size = tf.shape(images)
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])    # [N,C,H,W]
            flow_in = tf.transpose(flow_in, [0,3,1,2]) # [N,C,H,W]

        ## attention gating on raw images, assuming batch=2
        images = self._att_gate(images, atts)

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 16x
        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # Residual Block B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, images, 1, 'SAME', shape_dict['B0'], train_flag)

            # Pooling 1
            model['B0_pooled'] = nn.max_pool2d(self._data_format, model['B0'], 2, 'SAME')

            # Residual Block B1_0, B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'], train_flag)
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'], train_flag)

            # Pooling 2
            model['B1_2_pooled'] = nn.max_pool2d(self._data_format, model['B1_2'], 2, 'SAME')

            # Residual Block B2_0, B2_1, B2_2
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1, 1, 128, 256]
            shape_dict['B2']['convs'] = [[3, 3, 128, 256], [3, 3, 256, 256]]
            with tf.variable_scope('B2_0'):
                model['B2_0'] = nn.res_side(self._data_format, model['B1_2_pooled'], shape_dict['B2'], train_flag)
            for i in range(2):
                with tf.variable_scope('B2_' + str(i + 1)):
                    model['B2_' + str(i + 1)] = nn.res(self._data_format, model['B2_' + str(i)],
                                                       shape_dict['B2']['convs'], train_flag)

            # Pooling 3
            model['B2_2_pooled'] = nn.max_pool2d(self._data_format, model['B2_2'], 2, 'SAME')

            # Residual Block B3_0 - B3_5
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1, 1, 256, 512]
            shape_dict['B3']['convs'] = [[3, 3, 256, 512], [3, 3, 512, 512]]
            with tf.variable_scope('B3_0'):
                model['B3_0'] = nn.res_side(self._data_format, model['B2_2_pooled'], shape_dict['B3'], train_flag)
            for i in range(5):
                with tf.variable_scope('B3_' + str(i + 1)):
                    model['B3_' + str(i + 1)] = nn.res(self._data_format, model['B3_' + str(i)],
                                                       shape_dict['B3']['convs'], train_flag)

            # Pooling 4
            model['B3_5_pooled'] = nn.max_pool2d(self._data_format, model['B3_5'], 2, 'SAME')

            # Residual Block B4_0, B4_1, B4_2
            shape_dict['B4_0'] = {}
            shape_dict['B4_0']['side'] = [1, 1, 512, 1024]
            shape_dict['B4_0']['convs'] = [[3, 3, 512, 512],[3, 3, 512, 1024]]
            with tf.variable_scope('B4_0'):
                model['B4_0'] = nn.res_side(self._data_format, model['B3_5_pooled'], shape_dict['B4_0'], train_flag)
            shape_dict['B4_23'] = [[3, 3, 1024, 512], [3, 3, 512, 1024]]
            for i in range(2):
                with tf.variable_scope('B4_' + str(i + 1)):
                    model['B4_' + str(i + 1)] = nn.res(self._data_format, model['B4_' + str(i)],
                                                       shape_dict['B4_23'], train_flag)

            # aggregate all feature on diff levels for img(t+1)
            with tf.variable_scope('B1_side_path'):
                side_2 = nn.conv_layer(self._data_format, model['B1_2'][0:1,:,:,:], 1, 'SAME', [3, 3, 128, 16], train_flag)
                side_2 = nn.bias_layer(self._data_format, side_2, 16, train_flag)
                if self._data_format == "NCHW":
                    side_2 = tf.transpose(side_2, [0,2,3,1]) # To NHWC
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]]) # NHWC
                    side_2_f = tf.transpose(side_2_f, [0,3,1,2]) # To NCHW
                else:
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'][0:1,:,:,:], 1, 'SAME', [3, 3, 256, 16], train_flag)
                side_4 = nn.bias_layer(self._data_format, side_4, 16, train_flag)
                if self._data_format == "NCHW":
                    side_4 = tf.transpose(side_4, [0, 2, 3, 1])  # To NHWC
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
                    side_4_f = tf.transpose(side_4_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'][0:1,:,:,:], 1, 'SAME', [3, 3, 512, 16], train_flag)
                side_8 = nn.bias_layer(self._data_format, side_8, 16, train_flag)
                if self._data_format == "NCHW":
                    side_8 = tf.transpose(side_8, [0, 2, 3, 1])  # To NHWC
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
                    side_8_f = tf.transpose(side_8_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'][0:1,:,:,:], 1, 'SAME', [3, 3, 1024, 16], train_flag)
                side_16 = nn.bias_layer(self._data_format, side_16, 16, train_flag)
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
                main_feat_out = nn.conv_layer(self._data_format, concat_seg_feat, 1, 'SAME', [1, 1, 64, 64], train_flag)
                main_feat_out = nn.bias_layer(self._data_format, main_feat_out, 64, train_flag)

        with tf.variable_scope('feat_transform'):
            with tf.variable_scope('B0'):
                # feature map displacement using optical flow, original size
                B0_trans_feat = self.flow_disp(model['B0'][1:2,:,:,:], flow_in)
                # pool 1/4
                B0_trans_feat = nn.max_pool2d_4(self._data_format, B0_trans_feat, 4, 'SAME')
                # conv to resize feat channels
                shape_dict['B0_transform'] = [3, 3, 64, 256]
                with tf.variable_scope('conv_resize'):
                    model['B0_trans'] = nn.conv_layer(self._data_format, B0_trans_feat, 1, 'SAME',
                                                        shape_dict['B0_transform'], train_flag)
            with tf.variable_scope('B1'):
                # feature map displacement using optical flow, 1/2 size
                in_flow = nn.avg_pool2d(self._data_format, flow_in, 2, 2, 'SAME')
                B1_trans_feat = self.flow_disp(model['B1_2'][1:2, :, :, :], in_flow)
                # pool from 1/2 to 1/4
                B1_trans_feat = nn.max_pool2d(self._data_format, B1_trans_feat, 2, 'SAME')
                # conv to resize feat channels
                shape_dict['B1_transform'] = [3, 3, 128, 256]
                with tf.variable_scope('conv_resize'):
                    model['B1_trans'] = nn.conv_layer(self._data_format, B1_trans_feat, 1, 'SAME',
                                                      shape_dict['B1_transform'], train_flag)
            with tf.variable_scope('B2'):
                # feature map displacement using optical flow, 1/4 size
                in_flow = nn.avg_pool2d(self._data_format, flow_in, 4, 4, 'SAME')
                B2_trans_feat = self.flow_disp(model['B2_2'][1:2, :, :, :], in_flow)
                # no further pool needed, size 1/4, conv to resize feat channels
                shape_dict['B2_transform'] = [3, 3, 256, 256]
                with tf.variable_scope('conv_resize'):
                    model['B2_trans'] = nn.conv_layer(self._data_format, B2_trans_feat, 1, 'SAME',
                                                      shape_dict['B2_transform'], train_flag)
            with tf.variable_scope('B3'):
                # feature map displacement using optical flow, 1/8 size
                in_flow = nn.avg_pool2d(self._data_format, flow_in, 8, 8, 'SAME')
                B3_trans_feat = self.flow_disp(model['B3_5'][1:2, :, :, :], in_flow)
                # resize from 1/8 to 1/4, x2
                if self._data_format == "NCHW":
                    B3_trans_feat = tf.transpose(B3_trans_feat, [0, 2, 3, 1])  # To NHWC
                    B3_trans_feat = tf.image.resize_images(B3_trans_feat, [int(im_size[1]/4), int(im_size[2]/4)]) # NHWC
                    B3_trans_feat = tf.transpose(B3_trans_feat, [0, 3, 1, 2])  # To NCHW
                # conv to resize feat channels
                shape_dict['B3_transform'] = [3, 3, 512, 256]
                with tf.variable_scope('conv_resize'):
                    model['B3_trans'] = nn.conv_layer(self._data_format, B3_trans_feat, 1, 'SAME',
                                                      shape_dict['B3_transform'], train_flag)
            with tf.variable_scope('B4'):
                # feature map displacement using optical flow, 1/16 size
                in_flow = nn.avg_pool2d(self._data_format, flow_in, 16, 16, 'SAME')
                B4_trans_feat = self.flow_disp(model['B4_2'][1:2, :, :, :], in_flow)
                # resize from 1/16 to 1/4, x4
                if self._data_format == "NCHW":
                    B4_trans_feat = tf.transpose(B4_trans_feat, [0, 2, 3, 1])  # To NHWC
                    B4_trans_feat = tf.image.resize_images(B4_trans_feat, [int(im_size[1]/4), int(im_size[2]/4)]) # NHWC
                    B4_trans_feat = tf.transpose(B4_trans_feat, [0, 3, 1, 2])  # To NCHW
                # conv to resize feat channels
                shape_dict['B4_transform'] = [3, 3, 1024, 256]
                with tf.variable_scope('conv_resize'):
                    model['B4_trans'] = nn.conv_layer(self._data_format, B4_trans_feat, 1, 'SAME',
                                                      shape_dict['B4_transform'], train_flag)
            # concat all displaced features
            if self._data_format == "NCHW":
                concat_trans_feat = tf.concat([model['B0_trans'], model['B1_trans'], model['B2_trans'], model['B3_trans'], model['B4_trans']], axis=1)
            else:
                concat_trans_feat = tf.concat([model['B0_trans'], model['B1_trans'], model['B2_trans'], model['B3_trans'], model['B4_trans']], axis=3)
            # feat_trans_fuse, 3 convs
            with tf.variable_scope('fuse'):
                with tf.variable_scope('conv1'):
                    fuse1_out = nn.conv_layer(self._data_format, concat_trans_feat, 1, 'SAME',
                                              [3, 3, 1024, 512], train_flag)
                    fuse1_out = nn.bias_layer(self._data_format, fuse1_out, 512, train_flag)
                    fuse1_out = nn.ReLu_layer(fuse1_out)
                with tf.variable_scope('conv2'):
                    fuse2_out = nn.conv_layer(self._data_format, fuse1_out, 1, 'SAME',
                                              [3, 3, 512, 512], train_flag)
                    fuse2_out = nn.bias_layer(self._data_format, fuse2_out, 512, train_flag)
                    fuse2_out = nn.ReLu_layer(fuse2_out)
                with tf.variable_scope('conv3'):
                    fuse3_out = nn.conv_layer(self._data_format, fuse2_out, 1, 'SAME',
                                              [3, 3, 512, 512], train_flag)
                    fuse3_out = nn.bias_layer(self._data_format, fuse3_out, 512, train_flag)
                    fuse3_out = nn.ReLu_layer(fuse3_out)
                # resize to img_size and conv to 64 channels
                if self._data_format == "NCHW":
                    fuse3_out = tf.transpose(fuse3_out, [0, 2, 3, 1])  # To NHWC
                    fuse3_out = tf.image.resize_images(fuse3_out, [im_size[1], im_size[2]])  # NHWC
                    fuse3_out = tf.transpose(fuse3_out, [0, 3, 1, 2])  # To NCHW
                with tf.variable_scope('fuse_trans'):
                    trans_feat_out = nn.conv_layer(self._data_format, fuse3_out, 1, 'SAME', [1, 1, 512, 64],
                                                  train_flag)
                    trans_feat_out = nn.bias_layer(self._data_format, trans_feat_out, 64, train_flag)

        # classification loss
        with tf.variable_scope('classifier'):
            seg_feat = tf.concat([main_feat_out, trans_feat_out], axis=0) # [2,h,w,64] or [2,64,h,w]
            seg_out = nn.conv_layer(self._data_format, seg_feat, 1, 'SAME', [1, 1, 64, 2], train_flag)
            seg_out = nn.bias_layer(self._data_format, seg_out, 2, train_flag)


        return seg_feat, seg_out

    def flow_disp(self, feat_arr, flow_arr):
        '''
        :param feat_arr: [1,h,w,C] or [1,C,h,w]
        :param flow_arr: [1,h,w,2] or [1,2,h,w]
        :return: feat_arr: [1,h,w,C] or [1,C,h,w]
        '''

        if self._data_format == "NCHW":
            feat_arr = tf.transpose(feat_arr, [0, 2, 3, 1]) # to NHWC
            feat_arr = tf.squeeze(feat_arr, 0) # to HWC
            flow_arr = tf.transpose(flow_arr, [0, 2, 3, 1]) # to NHW2
            flow_arr = tf.squeeze(flow_arr, 0) # to HW2
        feat_shape = tf.shape(feat_arr)
        h = feat_shape[0]
        w = feat_shape[1]
        new_feat = tf.zeros_like(feat_arr)
        for idx_h in range(h):
            for idx_w in range(w):
                motion_h = int(round(flow_arr[idx_h][idx_w][1]))
                motion_w = int(round(flow_arr[idx_h][idx_w][0]))
                new_h = idx_h + motion_h
                new_w = idx_w + motion_w
                if new_h < h and new_h >= 0 and new_w < w and new_w >= 0:
                    new_feat[new_h][new_w] = feat_arr[idx_h][idx_w]
        # reshape
        if self._data_format == "NCHW":
            new_feat = tf.transpose(new_feat, [2, 0, 1]) # from HWC to CHW
            new_feat = tf.expand_dims(new_feat, 0) # to NCHW where N = 1

        return new_feat

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
                                         reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS) * 5.0
        tf.summary.scalar('sim_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, feed_att, flow_in, mask, weight, global_step, acc_count, train_flag):
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
        :param train_flag: 0:train main, 1:train trans, 2:train all, 3:no train all
        :return: total_loss, train_step_op, grad_acc_op
        '''
        seg_feat, seg_out = self._build_model(feed_img, feed_att, flow_in, train_flag) # [2,2,H,W] or [2,H,W,2]

        # total_loss:
        #   seg_loss for img(t+1)
        #   seg_loss for img(t)
        #   sim_loss between img(t+1), img(t)
        if train_flag == 2:
            total_loss = self._seg_loss(seg_out[0:1,:,:,:], feed_seg, feed_weight, feed_att[0:1,:,:,:], 'main') \
                        + self._seg_loss(seg_out[1:2,:,:,:], feed_seg, weight, mask, 'transformed') \
                        + self._sim_loss(seg_feat[1:2,:,:,:], seg_feat[0:1,:,:,:], feed_seg) \
                        + self._l2_loss(train_flag)
        elif train_flag == 1:
            total_loss = self._seg_loss(seg_out[1:2,:,:,:], feed_seg, weight, mask, 'transformed') \
                        + self._sim_loss(seg_feat[1:2,:,:,:], seg_feat[0:1,:,:,:], feed_seg) \
                        + self._l2_loss(train_flag)
        elif train_flag == 0:
            total_loss = self._seg_loss(seg_out[0:1,:,:,:], feed_seg, feed_weight, feed_att[0:1,:,:,:], 'main') \
                         + self._l2_loss(train_flag)
        else:
            sys.exit('No valid train_flag.')
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            seg_pred = seg_out
        if train_flag == 2:
            tf.summary.image('pred_main', tf.cast(tf.nn.softmax(seg_pred)[0:1, :, :, 1:2], tf.float16))
            tf.summary.image('pred_transformed', tf.cast(tf.nn.softmax(seg_pred)[1:2, :, :, 1:2], tf.float16))
        elif train_flag == 1:
            tf.summary.image('pred_transformed', tf.cast(tf.nn.softmax(seg_pred)[1:2, :, :, 1:2], tf.float16))
        elif train_flag == 0:
            tf.summary.image('pred_main', tf.cast(tf.nn.softmax(seg_pred)[0:1, :, :, 1:2], tf.float16))

        bp_step, grad_acc_op = self._optimize(total_loss, acc_count, global_step, train_flag)

        return total_loss, bp_step, grad_acc_op

    def _optimize(self, loss, acc_count, global_step, train_flag):
        '''
        :param loss: the network loss
        :return: a train op, a grad_acc_op
        '''

        optimizer = tf.train.AdamOptimizer(self._init_lr)
        grads_vars = optimizer.compute_gradients(loss)

        # create grad accumulator for each variable-grad pair
        grad_accumulator_0 = {}
        grad_accumulator_1 = {}
        grad_accumulator_2 = {}
        for idx in range(len(grads_vars)):
            if grads_vars[idx][0] is not None:
                var_name = str(grads_vars[idx][1].name).split(':')[0]
                if var_name.find('main') != -1:
                    grad_accumulator_0[idx] = tf.ConditionalAccumulator(grads_vars[idx][0].dtype)
                elif var_name.find('feat_transform') != -1:
                    grad_accumulator_1[idx] = tf.ConditionalAccumulator(grads_vars[idx][0].dtype)
                elif var_name.find('classifier') != -1:
                    grad_accumulator_2[idx] = tf.ConditionalAccumulator(grads_vars[idx][0].dtype)
        # apply gradient to each grad accumulator
        layer_lr = nn.param_lr()
        grad_accumulator_ops = []
        grad_accumulator_op0 = []
        grad_accumulator_op1 = []
        grad_accumulator_op2 = []
        for var_idx, grad_acc in grad_accumulator_0.iteritems():
            var_name = str(grads_vars[var_idx][1].name).split(':')[0]
            var_grad = tf.clip_by_value(grads_vars[var_idx][0], -1.0, 1.0)
            grad_accumulator_op0.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        for var_idx, grad_acc in grad_accumulator_1.iteritems():
            var_name = str(grads_vars[var_idx][1].name).split(':')[0]
            var_grad = tf.clip_by_value(grads_vars[var_idx][0], -1.0, 1.0)
            grad_accumulator_op1.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        for var_idx, grad_acc in grad_accumulator_2.iteritems():
            var_name = str(grads_vars[var_idx][1].name).split(':')[0]
            var_grad = tf.clip_by_value(grads_vars[var_idx][0], -1.0, 1.0)
            grad_accumulator_op2.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        grad_accumulator_ops.append(grad_accumulator_op0)
        grad_accumulator_ops.append(grad_accumulator_op1)
        grad_accumulator_ops.append(grad_accumulator_op2)
        # take average gradients for each variable after accumulating count reaches
        mean_grads_vars_0 = []
        mean_grads_vars_1 = []
        mean_grads_vars_2 = []
        for var_idx, grad_acc in grad_accumulator_0.iteritems():
            mean_grads_vars_0.append((grad_acc.take_grad(acc_count), grads_vars[var_idx][1]))
        for var_idx, grad_acc in grad_accumulator_1.iteritems():
            mean_grads_vars_1.append((grad_acc.take_grad(acc_count), grads_vars[var_idx][1]))
        for var_idx, grad_acc in grad_accumulator_2.iteritems():
            mean_grads_vars_2.append((grad_acc.take_grad(acc_count), grads_vars[var_idx][1]))

        # apply average gradients to variables
        if train_flag == 0:
            update_op = optimizer.apply_gradients(mean_grads_vars_0+mean_grads_vars_2, global_step=global_step)
        elif train_flag == 1:
            update_op = optimizer.apply_gradients(mean_grads_vars_1+mean_grads_vars_2, global_step=global_step)
        elif train_flag == 2:
            update_op = optimizer.apply_gradients(mean_grads_vars_0+mean_grads_vars_1+mean_grads_vars_2, global_step=global_step)

        return update_op, grad_accumulator_ops

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

    def _l2_loss(self, train_flag):

        l2_losses = []
        for var in tf.trainable_variables():
            # only regularize conv kernels
            if str(var.name).split(':')[0].find('bias') == -1:
                if train_flag == 0: # add main and classifier
                    if str(var.name).split(':')[0].find('main') != -1 or str(var.name).split(':')[0].find('classifier') != -1:
                        l2_losses.append(tf.nn.l2_loss(var))
                elif train_flag == 1: # add optical_flow, feat_transform and classifier
                    if str(var.name).split(':')[0].find('feat_transform') != -1 or \
                            str(var.name).split(':')[0].find('classifier') != -1:
                        l2_losses.append(tf.nn.l2_loss(var))
                elif train_flag == 2: # add all
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
