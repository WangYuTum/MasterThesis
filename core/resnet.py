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

    def _build_model(self, image):
        '''
        :param image: [1,H,W,3], tf.float32
        :return: last layer upsampled features
        '''
        model = {}

        im_size = tf.shape(image)
        if self._data_format == "NCHW":
            images = tf.transpose(image, [0,3,1,2])    # [N,C,H,W]

        ## The following 'main' scope is the primary (shared) feature layers, downsampling 8x
        shape_dict = {}
        shape_dict['B0'] = [3,3,3,64]

        with tf.variable_scope('main'):
            # B0
            with tf.variable_scope('B0'):
                model['B0'] = nn.conv_layer(self._data_format, image, 1, 'SAME', shape_dict['B0'])

            # Residual Block B1_0(down-sampling x2), B1_1, B1_2
            shape_dict['B1'] = {}
            shape_dict['B1']['side'] = [1, 1, 64, 128]
            shape_dict['B1']['convs'] = [[3, 3, 64, 128], [3, 3, 128, 128]]
            with tf.variable_scope('B1_0'):
                model['B1_0'] = nn.res_side(self._data_format, model['B0_pooled'], shape_dict['B1'])
            for i in range(2):
                with tf.variable_scope('B1_' + str(i + 1)):
                    model['B1_' + str(i + 1)] = nn.res(self._data_format, model['B1_' + str(i)],
                                                   shape_dict['B1']['convs'])

            # Residual Block B2_0(down-sampling x2), B2_1, B2_2
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


        return seg_feat

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
        :param train_flag: 0:train main, 1:train OF, 2:train all, 3:no train all
        :return: total_loss, train_step_op, grad_acc_op
        '''
        seg_feat, seg_out = self._build_model(feed_img, feed_att, flow_in, train_flag) # [2,2,H,W] or [2,H,W,2]

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
        grad_accumulator_0 = []
        grad_accumulator_1 = []
        grad_accumulator_2 = []
        for var_idx, grad_acc in grad_accumulator.iteritems():
            var_name = str(grads_vars[var_idx][1].name).split(':')[0]
            var_grad = grads_vars[var_idx][0]
            if var_name.find('main') != 0:
                grad_accumulator_0.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
            elif var_name.find('optical_flow') != 0 or var_name.find('feat_transform') != 0:
                grad_accumulator_1.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
            elif var_name.find('classifier') != 0:
                grad_accumulator_2.append(grad_acc.apply_grad(var_grad * layer_lr[var_name], local_step=global_step))
        grad_accumulator_op.append(grad_accumulator_0)
        grad_accumulator_op.append(grad_accumulator_1)
        grad_accumulator_op.append(grad_accumulator_2)
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
