'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

def conv_layer(data_format, input_tensor, stride=1, padding='SAME', shape=None):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    kernel = create_conv_kernel(shape)

    if data_format == "NCHW":
        conv_stride = [1,1,stride,stride]
    else:
        conv_stride = [1,stride,stride,1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)

    return conv_out

def res_side(data_format, input_tensor, shape_dict, is_train=False):
    ''' The residual block unit with side conv '''

    BN_out1 = BN(data_format, input_tensor, 'bn1', shape_dict['convs'][0][2], is_train)
    RELU_out1 = ReLu_layer(BN_out1)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, RELU_out1, 1, 'SAME', shape_dict['side'])
    # The 1st conv

def res(data_format, input_tensor, shape_dict, is_train=False):
    ''' The residual block unit with shortcut '''

    # TODO

def create_conv_kernel(shape=None):
    '''
    :param shape: the shape of kernel to be created
    :return: a tf.tensor
    '''
    # TODO: check if first created or restored
    init_op = tf.truncated_normal_initializer(stddev=0.001)
    var = tf.get_variable(name='kernel', shape=shape, initializer=init_op)

    return var

def max_pool2d(data_format, input_tensor, stride=2, padding='SAME'):
    ''' The standard max_pool2d with kernel size 2x2 '''

    if data_format == "NCHW":
        pool_size = [1,1,2,2]
        pool_stride = [1,1,stride,stride]
    else:
        pool_size = [1,2,2,1]
        pool_stride = [1,2,2,1]

    out = tf.nn.max_pool(input_tensor, pool_size, pool_stride, padding, data_format)

    return out

def BN(data_format, input_tensor, bn_scope=None, shape=None, is_train=False):

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
        norm_axis=1
    else:
        norm_axis=-1

    # TODO: check if tf.train.Saver restore variables of bn layer
    BN_out = tf.layers.batch_normalization(inputs=input_tensor,
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
                                           fused=bn_fused
                                           )

    return BN_out


def get_bn_params():
    # When first initialized/created use zero/one init, otherwise restore from .ckpt
    # TODO: check if first created or restored
    init_beta = tf.zeros_initializer()
    init_mean = tf.zeros_initializer()
    init_gamma = tf.ones_initializer()
    init_var = tf.ones_initializer()

    return init_beta, init_gamma, init_mean, init_var


def ReLu_layer(input_tensor):

    relu_out = tf.nn.relu(input_tensor)

    return relu_out



