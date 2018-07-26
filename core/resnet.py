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

    def _build_model(self, images, atts, bb, bb_mask, train_cnn):
        '''
        :param images: [1,H,W,3], tf.float32
        :return: segmentation output before softmax
        '''
        model = {}

        im_size = tf.shape(images)
        if self._data_format == "NCHW":
            images = tf.transpose(images, [0,3,1,2])    # [N,C,H,W]

        ## attention gating on raw images, assuming batch=1
        images = self._att_gate(images, atts)

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 16x
        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # Residual Block B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, images, 1, 'SAME', shape_dict['B0'], train_cnn)

            # Pooling 1
            model['B0_pooled'] = nn.max_pool2d(self._data_format, model['B0'], 2, 'SAME')

            # Residual Block B1_0, B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'], train_cnn)
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'], train_cnn)

            # Pooling 2
            model['B1_2_pooled'] = nn.max_pool2d(self._data_format, model['B1_2'], 2, 'SAME')

            # Residual Block B2_0, B2_1, B2_2
            shape_dict['B2'] = {}
            shape_dict['B2']['side'] = [1, 1, 128, 256]
            shape_dict['B2']['convs'] = [[3, 3, 128, 256], [3, 3, 256, 256]]
            with tf.variable_scope('B2_0'):
                model['B2_0'] = nn.res_side(self._data_format, model['B1_2_pooled'], shape_dict['B2'], train_cnn)
            for i in range(2):
                with tf.variable_scope('B2_' + str(i + 1)):
                    model['B2_' + str(i + 1)] = nn.res(self._data_format, model['B2_' + str(i)],
                                                       shape_dict['B2']['convs'], train_cnn)

            # Pooling 3
            model['B2_2_pooled'] = nn.max_pool2d(self._data_format, model['B2_2'], 2, 'SAME')

            # Residual Block B3_0 - B3_5
            shape_dict['B3'] = {}
            shape_dict['B3']['side'] = [1, 1, 256, 512]
            shape_dict['B3']['convs'] = [[3, 3, 256, 512], [3, 3, 512, 512]]
            with tf.variable_scope('B3_0'):
                model['B3_0'] = nn.res_side(self._data_format, model['B2_2_pooled'], shape_dict['B3'], train_cnn)
            for i in range(5):
                with tf.variable_scope('B3_' + str(i + 1)):
                    model['B3_' + str(i + 1)] = nn.res(self._data_format, model['B3_' + str(i)],
                                                       shape_dict['B3']['convs'], train_cnn)

            # Pooling 4
            model['B3_5_pooled'] = nn.max_pool2d(self._data_format, model['B3_5'], 2, 'SAME')

            # Residual Block B4_0, B4_1, B4_2
            shape_dict['B4_0'] = {}
            shape_dict['B4_0']['side'] = [1, 1, 512, 1024]
            shape_dict['B4_0']['convs'] = [[3, 3, 512, 512],[3, 3, 512, 1024]]
            with tf.variable_scope('B4_0'):
                model['B4_0'] = nn.res_side(self._data_format, model['B3_5_pooled'], shape_dict['B4_0'], train_cnn)
            shape_dict['B4_23'] = [[3, 3, 1024, 512], [3, 3, 512, 1024]]
            for i in range(2):
                with tf.variable_scope('B4_' + str(i + 1)):
                    model['B4_' + str(i + 1)] = nn.res(self._data_format, model['B4_' + str(i)],
                                                       shape_dict['B4_23'], train_cnn)

            shape_dict['feat_reduce'] = [1,1,1024,128]
            with tf.variable_scope('feat_reduce'):
                with tf.variable_scope('conv'):
                    model['feat_reduced'] = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME',
                                                        shape_dict['feat_reduce'], train_cnn)
                with tf.variable_scope('bias'):
                    model['feat_reduced'] = nn.bias_layer(self._data_format, model['feat_reduced'], 128, train_cnn)
                model['feat_reduced'] = nn.ReLu_layer(model['feat_reduced']) # [4,h,w,128] or [4,128,h,w]

                # resize to fixed size
                if self._data_format == "NCHW":
                    model['feat_reduced'] = tf.transpose(model['feat_reduced'], [0,2,3,1]) # To NHWC
                    model['feat_resized'] = tf.image.resize_images(model['feat_reduced'], [30, 56])
                    model['feat_resized'] = tf.transpose(model['feat_resized'], [0,3,1,2])  # To NCHW
                else:
                    model['feat_resized'] = tf.image.resize_images(model['feat_reduced'], [30, 56]) # NHWC

            ## The object descriptor branch, if not only train cnn part
            if not train_cnn:
                with tf.variable_scope('obj_desc'):
                    with tf.variable_scope('B4_up'):
                        side_b4 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 128], train_cnn)
                        side_b4 = nn.bias_layer(self._data_format, side_b4, 128, train_cnn)
                        if self._data_format == "NCHW":
                            side_b4 = tf.transpose(side_b4, [0, 2, 3, 1])  # To NHWC
                            side_b4_f = tf.image.resize_images(side_b4, [64, 128])  # NHWC
                            side_b4_f = tf.transpose(side_b4_f, [0, 3, 1, 2])  # To NCHW
                        else:
                            side_16_f = tf.image.resize_images(side_b4, [64, 128])  # NHWC
                    with tf.variable_scope('B5_up'):
                        side_b5 = nn.conv_layer(self._data_format, model['feat_resized'], 1, 'SAME', [3, 3, 128, 128], train_cnn)
                        side_b5 = nn.bias_layer(self._data_format, side_b5, 128, train_cnn)
                        if self._data_format == "NCHW":
                            side_b5 = tf.transpose(side_b5, [0, 2, 3, 1])  # To NHWC
                            side_b5_f = tf.image.resize_images(side_b5, [64, 128])  # NHWC
                            side_b5_f = tf.transpose(side_b5_f, [0, 3, 1, 2])  # To NCHW
                        else:
                            side_b5_f = tf.image.resize_images(side_b5, [64, 128])  # NHWC
                    with tf.variable_scope('concat_fuse'):
                        if self._data_format == "NCHW":
                            concat_obj_feat = tf.concat([side_b4_f, side_b5_f], axis=1)
                        else:
                            concat_obj_feat = tf.concat([side_b4_f], axis=3)
                        obj_feat = nn.conv_layer(self._data_format, concat_obj_feat, 1, 'SAME', [1, 1, 256, 256], train_cnn)
                        obj_feat = nn.bias_layer(self._data_format, obj_feat, 256, train_cnn)
                        obj_feat = nn.ReLu_layer(obj_feat)  # Add non-linearity, [N,64,128,256] or [N,256,64,128]
                    ## mask the relevant features, bb=[offset_h, offset_w, target_h, target_w]
                    ## bb_mask: [1, target_h, target_w, 1], tf.float32, zero-one tensor
                    ## NOTE: bb might be empty due to small object, set to minimum 2x2 size
                    if self._data_format == 'NCHW':
                        obj_feat = tf.transpose(obj_feat, [0, 2, 3, 1])  # To NHWC
                        masked_obj_feat = tf.image.crop_to_bounding_box(obj_feat, bb[0], bb[1], bb[2], bb[3])
                        masked_obj_feat = tf.multiply(masked_obj_feat, bb_mask) # [1,h,w,256]
                        masked_obj_feat = tf.image.resize_images(masked_obj_feat, [32, 32])  # NHWC
                        masked_obj_feat = tf.transpose(masked_obj_feat, [0, 3, 1, 2])  # To NCHW
                    else:
                        masked_obj_feat = tf.image.crop_to_bounding_box(obj_feat, bb[0], bb[1], bb[2], bb[3])
                        masked_obj_feat = tf.multiply(masked_obj_feat, bb_mask)  # [1,h,w,256]
                        masked_obj_feat = tf.image.resize_images(masked_obj_feat, [32, 32])  # NHWC

                    with tf.variable_scope('obj_feat_agg'):
                        with tf.variable_scope('conv1'):
                            obj_feat1 = nn.conv_layer(self._data_format, masked_obj_feat, 2, 'SAME', [3,3,256,256], train_cnn)
                            obj_feat1 = nn.bias_layer(self._data_format, obj_feat1, 256, train_cnn)
                            obj_feat1 = nn.ReLu_layer(obj_feat1) # [N,8,16,256] or [N,256,8,16]
                        with tf.variable_scope('conv2'):
                            obj_feat2 = nn.conv_layer(self._data_format, obj_feat1, 2, 'SAME', [3,3,256,256], train_cnn)
                            obj_feat2 = nn.bias_layer(self._data_format, obj_feat2, 256, train_cnn)
                            obj_feat2 = nn.ReLu_layer(obj_feat2) # [N,4,8,256] or [N,256,4,8]
                        # flatten to a vector
                        with tf.variable_scope('dense1'):
                            obj_vec = tf.layers.flatten(obj_feat2, name='obj_feat2_flat') # [N, 4*8*256] = [N, 8192]
                            dense_out1 = tf.layers.dense(obj_vec, 1024, activation=tf.nn.relu, use_bias=True) # [N,80]
                            dense_drop = tf.layers.dropout(dense_out1, rate=0.4, training=True)
                        with tf.variable_scope('dense2'):
                            dense_out2 = tf.layers.dense(dense_drop, 80, activation=None, use_bias=True)


            # aggregate all feature on diff levels
            with tf.variable_scope('B1_side_path'):
                side_2 = nn.conv_layer(self._data_format, model['B1_2'], 1, 'SAME', [3, 3, 128, 16], train_cnn)
                side_2 = nn.bias_layer(self._data_format, side_2, 16, train_cnn)
                if self._data_format == "NCHW":
                    side_2 = tf.transpose(side_2, [0,2,3,1]) # To NHWC
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]]) # NHWC
                    side_2_f = tf.transpose(side_2_f, [0,3,1,2]) # To NCHW
                else:
                    side_2_f = tf.image.resize_images(side_2, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('B2_side_path'):
                side_4 = nn.conv_layer(self._data_format, model['B2_2'], 1, 'SAME', [3, 3, 256, 16], train_cnn)
                side_4 = nn.bias_layer(self._data_format, side_4, 16, train_cnn)
                if self._data_format == "NCHW":
                    side_4 = tf.transpose(side_4, [0, 2, 3, 1])  # To NHWC
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
                    side_4_f = tf.transpose(side_4_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_4_f = tf.image.resize_images(side_4, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B3_side_path'):
                side_8 = nn.conv_layer(self._data_format, model['B3_5'], 1, 'SAME', [3, 3, 512, 16], train_cnn)
                side_8 = nn.bias_layer(self._data_format, side_8, 16, train_cnn)
                if self._data_format == "NCHW":
                    side_8 = tf.transpose(side_8, [0, 2, 3, 1])  # To NHWC
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
                    side_8_f = tf.transpose(side_8_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_8_f = tf.image.resize_images(side_8, [im_size[1], im_size[2]]) # NHWC
            with tf.variable_scope('B4_side_path'):
                side_16 = nn.conv_layer(self._data_format, model['B4_2'], 1, 'SAME', [3, 3, 1024, 16], train_cnn)
                side_16 = nn.bias_layer(self._data_format, side_16, 16, train_cnn)
                if self._data_format == "NCHW":
                    side_16 = tf.transpose(side_16, [0, 2, 3, 1])  # To NHWC
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]]) # NHWC
                    side_16_f = tf.transpose(side_16_f, [0, 3, 1, 2])  # To NCHW
                else:
                    side_16_f = tf.image.resize_images(side_16, [im_size[1], im_size[2]])  # NHWC
            with tf.variable_scope('resize_side_path'):
                side_reduced = nn.conv_layer(self._data_format, model['feat_resized'], 1, 'SAME', [3, 3, 128, 16], train_cnn)
                side_reduced = nn.bias_layer(self._data_format, side_reduced, 16, train_cnn)
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
                seg_out = nn.conv_layer(self._data_format, concat_seg_feat, 1, 'SAME', [1, 1, 80, 2], train_cnn)
                init_seg_out = nn.bias_layer(self._data_format, seg_out, 2, train_cnn)

            ## Fuse multi-scale features with object descriptor, if not train only cnn part
            if not train_cnn:
                with tf.variable_scope('obj_fuse'):
                    if self._data_format == "NCHW":
                        fused_glob_seg = tf.transpose(concat_seg_feat, [0, 2, 3, 1])  # To NHWC
                        fused_glob_seg = tf.multiply(fused_glob_seg, dense_out2)  # NHWC
                        fused_glob_seg = tf.transpose(fused_glob_seg, [0, 3, 1, 2])  # To NCHW
                    else:
                        fused_glob_seg = tf.multiply(concat_seg_feat, dense_out2)
                    fused_seg = nn.conv_layer(self._data_format, fused_glob_seg, 1, 'SAME', [1, 1, 80, 2], train_cnn)
                    final_seg_out = nn.bias_layer(self._data_format, fused_seg, 2, train_cnn)
                    final_out = final_seg_out
            else:
                final_out = init_seg_out

        return final_out

    def _att_gate(self, images, atts):
        '''
        :param images: [1,H,W,3] or [1,3,H,W], tf.float32
        :param atts: [1,H,W,1], tf.int32
        :return: gated image
        '''
        if self._data_format == 'NCHW':
            atts = tf.transpose(atts, [0, 3, 1, 2])
        att_mask = tf.cast(atts, tf.float32)
        gated_img = tf.multiply(images, att_mask)

        return gated_img

    def _seg_loss(self, seg_out, seg_gt, seg_weight, att):
        '''
        :param seg_out: logits, [4,H,W,2] or [4,2,H,W], tf.float32
        :param seg_gt: [4,H,W,1], tf.int32
        :param seg_weight: [4,H,W,1], tf.float32
        :return: scalar (balanced cross-entropy), tf.float32
        '''

        loss = self._balanced_cross_entropy(input_tensor=seg_out,
                                            labels=seg_gt,
                                            weight=seg_weight,
                                            att=att)
        tf.summary.scalar('seg_loss', loss)

        return loss

    def train(self, feed_img, feed_seg, feed_weight, feed_att, bb, bb_mask, global_step, acc_count, train_cnn):
        '''
        :param feed_img: [1,H,W,3], tf.float32
        :param feed_seg: [1,H,W,1], tf.int32
        :param feed_weight: [1,H,W,1], tf.float32
        :param feed_att: [1,H,W,1], tf.int32
        :param bb: a list containing [offset_h, offset_w, target_h, target_w], tf.int32
        :param bb_mask: zero-one tensor [1,target_h, target_w, 1], tf.float32
        :param global_step: keep track of global train step
        :param acc_count: number of accumulated gradients
        :param train_cnn: bool, train only cnn or train object descriptor
        :return: total_loss, train_step_op, grad_acc_op
        '''

        seg_out = self._build_model(feed_img, feed_att, bb, bb_mask, train_cnn) # seg_out shape: [1,H,W,2] or [1,2,H,W] with original input image size
        total_loss = self._seg_loss(seg_out, feed_seg, feed_weight, feed_att) \
                     + self._l2_loss()
        tf.summary.scalar('total_loss', total_loss)

        # display current output
        if self._data_format == "NCHW":
            seg_pred = tf.transpose(seg_out, [0,2,3,1])
        else:
            seg_pred = seg_out
        tf.summary.image('pred', tf.cast(tf.nn.softmax(seg_pred)[:, :, :, 1:2], tf.float16))

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
            # do not regularize bias
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
