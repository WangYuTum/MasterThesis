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

    def _build_model(self, images, atts, prob, bb, training=False, feed_state=None):
        '''
        :param images: [8,H,W,3], tf.float32
        :param atts: [8,H,W,1], tf.int32
        :param prob: [8,H,W,1], tf.float32
        :param bb: a list of length 8, each element is a list of length 4 specifying [offset_h, offset_w, target_h, target_w]
        :param training: bool
        :param feed_state: only used in inf for now, tuple ([1,256,512,64],[1,256,512,64]), tf.float32
        :return: segmentation output before softmax
        '''
        model = {}

        im_size = tf.shape(images)
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])    # [N,C,H,W]

        ## attention gating on raw images, assuming seq=2, frames=4, hence batch=8
        images = self._att_gate(images, atts)

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 16x
        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # Residual Block B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, images, 1, 'SAME', shape_dict['B0'], training)

            # Pooling 1
            model['B0_pooled'] = nn.max_pool2d(self._data_format, model['B0'], 2, 'SAME')

            # Residual Block B1_0, B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'], training)
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'], training)

            # Pooling 2
            model['B1_2_pooled'] = nn.max_pool2d(self._data_format, model['B1_2'], 2, 'SAME')

            # Residual Block B2_0, B2_1, B2_2
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1, 1, 128, 256]
            shape_dict['B2']['convs'] = [[3, 3, 128, 256], [3, 3, 256, 256]]
            with tf.variable_scope('B2_0'):
                model['B2_0'] = nn.res_side(self._data_format, model['B1_2_pooled'], shape_dict['B2'], training)
            for i in range(2):
                with tf.variable_scope('B2_' + str(i + 1)):
                    model['B2_' + str(i + 1)] = nn.res(self._data_format, model['B2_' + str(i)],
                                                       shape_dict['B2']['convs'], training)

            # Pooling 3
            model['B2_2_pooled'] = nn.max_pool2d(self._data_format, model['B2_2'], 2, 'SAME')

            # Residual Block B3_0 - B3_5
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1, 1, 256, 512]
            shape_dict['B3']['convs'] = [[3, 3, 256, 512], [3, 3, 512, 512]]
            with tf.variable_scope('B3_0'):
                model['B3_0'] = nn.res_side(self._data_format, model['B2_2_pooled'], shape_dict['B3'], training)
            for i in range(5):
                with tf.variable_scope('B3_' + str(i + 1)):
                    model['B3_' + str(i + 1)] = nn.res(self._data_format, model['B3_' + str(i)],
                                                       shape_dict['B3']['convs'], training)

            # Pooling 4
            model['B3_5_pooled'] = nn.max_pool2d(self._data_format, model['B3_5'], 2, 'SAME')

            # Residual Block B4_0, B4_1, B4_2
            shape_dict['B4_0'] = {}
            shape_dict['B4_0']['side'] = [1, 1, 512, 1024]
            shape_dict['B4_0']['convs'] = [[3, 3, 512, 512],[3, 3, 512, 1024]]
            with tf.variable_scope('B4_0'):
                model['B4_0'] = nn.res_side(self._data_format, model['B3_5_pooled'], shape_dict['B4_0'], training)
            shape_dict['B4_23'] = [[3, 3, 1024, 512], [3, 3, 512, 1024]]
            for i in range(2):
                with tf.variable_scope('B4_' + str(i + 1)):
                    model['B4_' + str(i + 1)] = nn.res(self._data_format, model['B4_' + str(i)],
                                                       shape_dict['B4_23'], training)

            # aggregate all feature on diff levels
            with tf.variable_scope('B1_side_path'):
                side_2 = nn.conv_layer(self._data_format, model['B1_2'], 1, 'SAME', [3, 3, 128, 16], training)
                side_2 = nn.bias_layer(self._data_format, side_2, 16, training)
                if self._data_format == "NCHW":
                    side_2 = tf.transpose(side_2, [0,2,3,1]) # To NHWC
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]]) # NHWC
                    side_2_f = tf.transpose(side_2_f, [0,3,1,2]) # To NCHW
                else:
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'], 1, 'SAME', [3, 3, 256, 16], training)
                side_4 = nn.bias_layer(self._data_format, side_4, 16, training)
                if self._data_format == "NCHW":
                    side_4 = tf.transpose(side_4, [0, 2, 3, 1])  # To NHWC
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
                    side_4_f = tf.transpose(side_4_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'], 1, 'SAME', [3, 3, 512, 16], training)
                side_8 = nn.bias_layer(self._data_format, side_8, 16, training)
                if self._data_format == "NCHW":
                    side_8 = tf.transpose(side_8, [0, 2, 3, 1])  # To NHWC
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
                    side_8_f = tf.transpose(side_8_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 16], training)
                side_16 = nn.bias_layer(self._data_format, side_16, 16, training)
                if self._data_format == "NCHW":
                    side_16 = tf.transpose(side_16, [0, 2, 3, 1])  # To NHWC
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]]) # NHWC
                    side_16_f = tf.transpose(side_16_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]])  # NHWC

            # concat
            if self._data_format == "NCHW":
                concat_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=1) # [N, 64, H, W]
            else:
                concat_feat = tf.concat([side_2_f, side_4_f, side_8_f, side_16_f], axis=3) # [N, H, W, 64]

        # only in inf mode, use branch to estimate init prob_map
            if not training:
                with tf.variable_scope('fuse'):
                    init_out = nn.conv_layer(self._data_format, concat_feat, 1, 'SAME', [1, 1, 64, 2])
                    init_out = nn.bias_layer(self._data_format, init_out, 2)

        # feature update branch
        # TODO: adapt code to use batch = 1, 8, 16
        with tf.variable_scope('feat_update'):
            # get bbox from att box coordinates, assuming batch=8
            if training:
                feats = [0] * 10
                if self._data_format == 'NCHW':
                    concat_feat = tf.transpose(concat_feat, [0, 2, 3, 1])  # To [N, H, W, 64]
                for i in range(10):
                    sub_feats = tf.image.crop_to_bounding_box(concat_feat[i:i+1,:,:,:], bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                    sub_att = tf.image.crop_to_bounding_box(tf.cast(atts[i:i+1,:,:,:], tf.float32), bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                    sub_conf = tf.image.crop_to_bounding_box(prob[i:i+1,:,:,:], bb[i][0], bb[i][1], bb[i][2], bb[i][3])
                    sub_att_feat = tf.multiply(sub_feats, sub_att)
                    sub_att_feat = tf.concat([sub_conf, sub_att_feat], axis=3) # [1, H, W, 65]
                    feats[i] = tf.image.resize_images(sub_att_feat, [256, 512])  # [1, 256, 512, 65]
                # stack all boxed feats to batch=10
                stack_feats = tf.concat([feats[0], feats[1], feats[2], feats[3],
                                         feats[4], feats[5], feats[6], feats[7],
                                         feats[8], feats[9]], axis=0) # [10, 256, 512, 65]
            else: # in test mode, batch=1
                if self._data_format == 'NCHW':
                    concat_feat = tf.transpose(concat_feat, [0, 2, 3, 1])  # To [1, H, W, 64]
                sub_feat = tf.image.crop_to_bounding_box(concat_feat, bb[0], bb[1], bb[2], bb[3])
                sub_att = tf.image.crop_to_bounding_box(tf.cast(atts, tf.float32), bb[0], bb[1], bb[2], bb[3])
                sub_conf = tf.image.crop_to_bounding_box(prob, bb[0], bb[1], bb[2], bb[3])
                sub_att_feat = tf.multiply(sub_feat, sub_att)
                sub_att_feat = tf.concat([sub_conf, sub_att_feat], axis=3)  # [1, H, W, 65]
                stack_feats = tf.image.resize_images(sub_att_feat, [256, 512])  # [1, 256, 512, 65]
                init_state_inf = stack_feats[:,:,:,1:65]
            if self._data_format == 'NCHW':
                stack_feats = tf.transpose(stack_feats, [0, 3, 1, 2]) # [N, 65, H, W]
            # go through a conv 1x1 before lstm
            with tf.variable_scope('conv_in_lstm'):
                feat_fuse = nn.conv_layer(self._data_format, stack_feats, 1, 'SAME', [1, 1, 65, 65], training)
                feat_fuse = nn.bias_layer(self._data_format, feat_fuse, 65, training)
                feat_fuse = nn.ReLu_layer(feat_fuse)
            # LSTM layer
            with tf.variable_scope('lstm_2d'):
                if training:
                    lstm_out = nn.lstm_conv2d_train(self._data_format, feat_fuse) # out: [10, 64, 256, 512] if NCHW
                else:
                    lstm_out, current_state, assign_state_ops = nn.lstm_conv2d_inf(self._data_format,
                                                                                   feat_fuse,
                                                                                   feed_state=feed_state)
                    # lstm out: [1, 64, 256, 512] if NCHW, state_tuple: ([1,256,512,64], [1,256,512,64])
            # resize all bbox to original size and pad to image size
            if training:
                full_feats = [0] * 10
                if self._data_format == 'NCHW':
                    lstm_out = tf.transpose(lstm_out, [0, 2, 3, 1]) # to [8, 256, 512, 64]
                for i in range(10):
                    full_feat = tf.image.resize_images(lstm_out[i:i+1,:,:,:], [bb[i][2], bb[i][3]]) # to bbox original size
                    full_feats[i] = tf.image.pad_to_bounding_box(full_feat, bb[i][0], bb[i][1], im_size[1], im_size[2]) # to input image size
                stack_full_feats = tf.concat([full_feats[0], full_feats[1], full_feats[2], full_feats[3],
                                              full_feats[4], full_feats[5], full_feats[6], full_feats[7],
                                              full_feats[8], full_feats[9]], axis=0)
            else: # in test mode, batch=1
                if self._data_format == 'NCHW':
                    lstm_out = tf.transpose(lstm_out, [0, 2, 3, 1]) # to [8, 256, 512, 64]
                full_feat = tf.image.resize_images(lstm_out, [bb[2], bb[3]])  # to bbox original size
                stack_full_feats = tf.image.pad_to_bounding_box(full_feat, bb[0], bb[1], im_size[1], im_size[2]) # to input image size
            if self._data_format == 'NCHW':
                stack_full_feats = tf.transpose(stack_full_feats, [0, 3, 1, 2]) # [8, H, W, 64]
            # conv after lstm
            with tf.variable_scope('conv_out_lstm'):
                feat_after_lstm = nn.conv_layer(self._data_format, stack_full_feats, 1, 'SAME', [1, 1, 64, 64], training)
                feat_after_lstm = nn.bias_layer(self._data_format, feat_after_lstm, 64, training)
                feat_after_lstm = nn.ReLu_layer(feat_after_lstm)
            # conv classifier
            with tf.variable_scope('conv_cls'):
                seg_out = nn.conv_layer(self._data_format, feat_after_lstm, 1, 'SAME', [1, 1, 64, 2], training)
                seg_out = nn.bias_layer(self._data_format, seg_out, 2, training)

        if training:
            return seg_out
        else:
            return seg_out, init_out, current_state, assign_state_ops, init_state_inf

    def _att_gate(self, images, atts):
        '''
        :param images: [8,H,W,3] or [8,3,H,W], tf.float32
        :param atts: [8,H,W,1], tf.int32
        :return: gated image
        '''
        if self._data_format == 'NCHW':
            atts = tf.transpose(atts, [0, 3, 1, 2])
        att_mask = tf.cast(atts, tf.float32)
        gated_img = tf.multiply(images, att_mask) # automatically broadcasting

        return gated_img

    def _seg_loss(self, seg_out, seg_gt, seg_weight, att):
        '''
        :param seg_out: logits, [8,H,W,2] or [8,2,H,W], tf.float32
        :param seg_gt: [8,H,W,1], tf.int32
        :param seg_weight: [8,H,W,1], tf.float32
        :return: scalar (balanced cross-entropy), tf.float32
        '''

        loss = self._balanced_cross_entropy(input_tensor=seg_out,
                                            labels=seg_gt,
                                            weight=seg_weight,
                                            att=att)
        tf.summary.scalar('seg_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, feed_att, feed_prob, bb, global_step):
        '''
        :param feed_img: [8,H,W,3], tf.float32
        :param feed_seg: [8,H,W,1], tf.int32
        :param feed_weight: [8,H,W,1], tf.float32
        :param feed_att: [8,H,W,1], tf.int32
        :param feed_prob: [8,H,W,1], tf.float32
        :param bb: bbox params for batch=8
        :param global_step: keep track of global train step
        :return: total_loss, bp_step
        '''

        seg_out = self._build_model(feed_img, feed_att, feed_prob, bb, True) # original image size
        total_loss = self._seg_loss(seg_out, feed_seg, feed_weight, feed_att) \
                     + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            seg_pred = seg_out
        tf.summary.image('pred', tf.cast(tf.nn.softmax(seg_pred)[:, :, :, 1:2], tf.float16))

        bp_step = tf.train.AdamOptimizer(self._init_lr).minimize(total_loss, global_step=global_step)

        return total_loss, bp_step

    def test(self, feed_img, feed_att, feed_prob, feed_bb, feed_state):
        '''
        :param feed_img: [1, H, W, 3], tf.float32
        :param feed_att: [1, H, W, 1], tf.int32
        :param feed_prob: [1, H, W, 1], tf.float32, obtained from init_out
        :param feed_bb: [4], tf.int32, obtained from feed_att
        :param feed_state: ([1,256,512,64],[1,256,512,64]), tf.float32, obtained from previous frame
        :return:
            - 1st forward pass run: init_prob_map, init_seg_mask
            - 2nd forward pass run: 1) assign_state_ops, 2) final_seg_mask, current_state
        '''
        final_out, init_out, current_state, assign_state_ops, init_lstm_inf = self._build_model(feed_img, feed_att, feed_prob, feed_bb, False, feed_state)

        # process init_out
        if self._data_format == "NCHW":
            init_out = tf.transpose(init_out, [0, 2, 3, 1])
        init_prob_map = tf.nn.softmax(init_out) # [1, H, W, 2]
        init_seg_mask = tf.argmax(init_prob_map, axis=3, output_type=tf.int32)  # [1, H, W]

        # process final_out
        if self._data_format == "NCHW":
            final_out = tf.transpose(final_out, [0, 2, 3, 1])
        final_prob_map = tf.nn.softmax(final_out) # [1, H, W, 2]
        final_seg_mask = tf.argmax(final_prob_map, axis=3, output_type=tf.int32)  # [1, H, W]


        return init_prob_map[:,:,:,1:], init_seg_mask, final_seg_mask, assign_state_ops, current_state, init_lstm_inf

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
