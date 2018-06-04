'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys

def conv_layer(data_format, input_tensor, stride=1, padding='SAME', shape=None):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    kernel = create_conv_kernel(shape)

    if data_format == "NCHW":
        conv_stride = [1, 1, stride, stride]
    else:
        conv_stride = [1, stride, stride, 1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)

    return conv_out


def res_side(data_format, input_tensor, shape_dict, is_train=False):
    ''' The residual block unit with side conv '''

    # The # The 1st bn layer
    bn_out1 = BN(data_format, input_tensor, 'bn1', is_train)
    relu_out1 = ReLu_layer(bn_out1)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['side'])
    # The 1st conv
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['convs'][0])
    # The 2nd bn layer
    bn_out2 = BN(data_format, conv_out1, 'bn2', is_train)
    relu_out2 = ReLu_layer(bn_out2)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_dict['convs'][1])
    # Fuse
    block_out = tf.add(side_out, conv_out2)

    return  block_out


def res(data_format, input_tensor, shape_dict, is_train=False):
    ''' The residual block unit with shortcut '''

    scope_name = tf.get_variable_scope().name
    if scope_name.find('B4') != -1:
        shape_conv1 = shape_dict[0]
        shape_conv2 = shape_dict[1]
    else:
        shape_conv1 = shape_dict[1]
        shape_conv2 = shape_dict[1]

    # The 1st bn layer
    bn_out1 = BN(data_format, input_tensor, 'bn1', is_train)
    relu_out1 = ReLu_layer(bn_out1)
    # The 1st conv layer
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_conv1)
    # The 2nd bn layer
    bn_out2 = BN(data_format, conv_out1, 'bn2', is_train)
    relu_out2 = ReLu_layer(bn_out2)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_conv2)
    # Fuse
    block_out = tf.add(input_tensor, conv_out2)

    return block_out


def BN(data_format, input_tensor, bn_scope=None, is_train=False):

    scope_name = tf.get_variable_scope().name + '/' + bn_scope
    print('Layer name: %s'%scope_name)

    [beta_init, gamma_init, mean_init, var_init] = get_bn_params()
    bn_training = is_train
    bn_trainable = True
    bn_fused = False

    if is_train is False:
        bn_training = False
        bn_trainable = False
        bn_fused = True
    else:
        bn_training = True
        bn_trainable = True
        bn_fused = False

    if data_format == "NCHW":
        norm_axis = 1
    else:
        norm_axis = -1

    bn_out = tf.layers.batch_normalization(inputs=input_tensor,
                                           axis=norm_axis,
                                           momentum=0.99,
                                           epsilon=0.001,
                                           center=True,
                                           scale=True,
                                           beta_initializer=beta_init,
                                           gamma_initializer=gamma_init,
                                           moving_mean_initializer=mean_init,
                                           moving_variance_initializer=var_init,
                                           beta_regularizer=None,
                                           gamma_regularizer=None,
                                           training=bn_training,
                                           trainable=bn_trainable,
                                           name=bn_scope,
                                           reuse=None,
                                           fused=bn_fused)

    return bn_out


def get_bn_params():
    # When first initialized/created use zero/one init, otherwise restore from .ckpt

    init_beta = tf.zeros_initializer()
    init_mean = tf.zeros_initializer()
    init_gamma = tf.ones_initializer()
    init_var = tf.ones_initializer()

    return init_beta, init_gamma, init_mean, init_var


def create_conv_kernel(shape=None):
    '''
    :param shape: the shape of kernel to be created
    :return: a tf.tensor
    '''
    init_op = tf.truncated_normal_initializer(stddev=0.001)
    var = tf.get_variable(name='kernel', shape=shape, initializer=init_op)

    return var


def max_pool2d(data_format, input_tensor, stride=2, padding='SAME'):
    ''' The standard max_pool2d with kernel size 2x2 '''

    if data_format == "NCHW":
        pool_size = [1, 1, 2, 2]
        pool_stride = [1, 1, stride, stride]
    else:
        pool_size = [1, 2, 2, 1]
        pool_stride = [1, 2, 2, 1]

    out = tf.nn.max_pool(input_tensor, pool_size, pool_stride, padding, data_format)

    return out


def ReLu_layer(input_tensor):

    relu_out = tf.nn.relu(input_tensor)

    return relu_out


def bias_layer(data_format, input_tensor, shape=None):

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    bias = create_bias(shape)
    bias_out = tf.nn.bias_add(input_tensor, bias, data_format)

    return bias_out

def create_bias(shape=None):

    init = tf.zeros_initializer()
    var = tf.get_variable('bias', initializer=init, shape=shape)

    return var


def get_imgnet_var():

    ## Imgnet weights variables
    imgnet_dict = {}
    # for the first conv
    with tf.variable_scope('main/B0', reuse=True):
        imgnet_dict['main/B0/kernel'] = tf.get_variable('kernel')
    # for all resnet side convs
    for i in range(4):
        with tf.variable_scope('main/B' + str(i + 1) + '_0/side', reuse=True):
            imgnet_dict['main/B' + str(i + 1) + '_0/side/kernel'] = tf.get_variable('kernel')
    # for all convs on the main path
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv1', reuse=True):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel'] = tf.get_variable('kernel')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv2', reuse=True):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel'] = tf.get_variable('kernel')
    # for convs on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv1', reuse=True):
            imgnet_dict['main/B3_' + str(i + 3) + '/conv1/kernel'] = tf.get_variable('kernel')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv2', reuse=True):
            imgnet_dict['main/B3_' + str(i + 3) + '/conv2/kernel'] = tf.get_variable('kernel')
    # for all batchnorm layers
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/bn1', reuse=True):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn1/beta'] = tf.get_variable('beta')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn1/gamma'] = tf.get_variable('gamma')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn1/moving_mean'] = tf.get_variable('moving_mean')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn1/moving_variance'] = tf.get_variable('moving_variance')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) +  '/bn2', reuse=True):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn2/beta'] = tf.get_variable('beta')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn2/gamma'] = tf.get_variable('gamma')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn2/moving_mean'] = tf.get_variable('moving_mean')
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/bn2/moving_variance'] = tf.get_variable('moving_variance')
    # for batchnorm on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/bn1', reuse=True):
            imgnet_dict['main/B3_' + str(i + 3) + '/bn1/beta'] = tf.get_variable('beta')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn1/gamma'] = tf.get_variable('gamma')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn1/moving_mean'] = tf.get_variable('moving_mean')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn1/moving_variance'] = tf.get_variable('moving_variance')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/bn2', reuse=True):
            imgnet_dict['main/B3_' + str(i + 3) + '/bn2/beta'] = tf.get_variable('beta')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn2/gamma'] = tf.get_variable('gamma')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn2/moving_mean'] = tf.get_variable('moving_mean')
            imgnet_dict['main/B3_' + str(i + 3) + '/bn2/moving_variance'] = tf.get_variable('moving_variance')

    return imgnet_dict


def param_lr():
    '''
    Set relative learning rate for different layers. The final lr is the global lr multiplied by the relative rate.
    :return: A dict key: var_name, value: relative rate
    '''
    vars_lr = dict()
    
    vars_lr['main/B0/kernel'] = 1.0

    vars_lr['main/B1_0/side/kernel'] = 1.0
    vars_lr['main/B1_0/conv1/kernel'] = 1.0
    vars_lr['main/B1_0/conv2/kernel'] = 1.0
    vars_lr['main/B1_1/conv1/kernel'] = 1.0
    vars_lr['main/B1_1/conv2/kernel'] = 1.0
    vars_lr['main/B1_2/conv1/kernel'] = 1.0
    vars_lr['main/B1_2/conv2/kernel'] = 1.0

    vars_lr['main/B2_0/side/kernel'] = 1.0
    vars_lr['main/B2_0/conv1/kernel'] = 1.0
    vars_lr['main/B2_0/conv2/kernel'] = 1.0
    vars_lr['main/B2_1/conv1/kernel'] = 1.0
    vars_lr['main/B2_1/conv2/kernel'] = 1.0
    vars_lr['main/B2_2/conv1/kernel'] = 1.0
    vars_lr['main/B2_2/conv2/kernel'] = 1.0

    vars_lr['main/B3_0/side/kernel'] = 1.0
    vars_lr['main/B3_0/conv1/kernel'] = 1.0
    vars_lr['main/B3_0/conv2/kernel'] = 1.0
    vars_lr['main/B3_1/conv1/kernel'] = 1.0
    vars_lr['main/B3_1/conv2/kernel'] = 1.0
    vars_lr['main/B3_2/conv1/kernel'] = 1.0
    vars_lr['main/B3_2/conv2/kernel'] = 1.0
    vars_lr['main/B3_3/conv1/kernel'] = 1.0
    vars_lr['main/B3_3/conv2/kernel'] = 1.0
    vars_lr['main/B3_4/conv1/kernel'] = 1.0
    vars_lr['main/B3_4/conv2/kernel'] = 1.0
    vars_lr['main/B3_5/conv1/kernel'] = 1.0
    vars_lr['main/B3_5/conv2/kernel'] = 1.0

    vars_lr['main/B4_0/side/kernel'] = 1.0
    vars_lr['main/B4_0/conv1/kernel'] = 1.0
    vars_lr['main/B4_0/conv2/kernel'] = 1.0
    vars_lr['main/B4_1/conv1/kernel'] = 1.0
    vars_lr['main/B4_1/conv2/kernel'] = 1.0
    vars_lr['main/B4_2/conv1/kernel'] = 1.0
    vars_lr['main/B4_2/conv2/kernel'] = 1.0

    vars_lr['main/feat_reduce/conv/kernel'] = 1.0
    vars_lr['main/feat_reduce/bias/bias'] = 2.0

    vars_lr['main/B1_side_path/kernel'] = 1.0
    vars_lr['main/B1_side_path/bias'] = 2.0
    vars_lr['main/B2_side_path/kernel'] = 1.0
    vars_lr['main/B2_side_path/bias'] = 2.0
    vars_lr['main/B3_side_path/kernel'] = 1.0
    vars_lr['main/B3_side_path/bias'] = 2.0
    vars_lr['main/B4_side_path/kernel'] = 1.0
    vars_lr['main/B4_side_path/bias'] = 2.0
    vars_lr['main/resize_side_path/kernel'] = 1.0
    vars_lr['main/resize_side_path/bias'] = 2.0

    vars_lr['main/fuse/kernel'] = 0.01
    vars_lr['main/fuse/bias'] = 0.02

    return vars_lr









