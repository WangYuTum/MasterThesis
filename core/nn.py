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
        conv_stride = [1,1,stride,stride]
    else:
        conv_stride = [1,stride,stride,1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)

    return conv_out


def res_side(data_format, input_tensor, shape_dict, is_train=False):
    ''' The residual block unit with side conv '''

    # The 1st bn layer
    BN_out1 = BN(data_format, input_tensor, 'bn1', shape_dict['convs'][0][2], is_train)
    RELU_out1 = ReLu_layer(BN_out1)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, RELU_out1, 1, 'SAME', shape_dict['side'])
    # The 1st conv
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, 1, 'SAME', shape_dict['convs'][0])
    # The 2nd bn layer
    BN_out2 = BN(data_format, CONV_out1, 'bn2', shape_dict['convs'][1][2], is_train)
    RELU_out2 = ReLu_layer(BN_out2)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, 1, 'SAME', shape_dict['convs'][1])
    # Fuse
    RES_out = tf.add(side_out, CONV_out2)

    return  RES_out


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
    BN_out1 = BN(data_format, input_tensor, 'bn1', shape_conv1[2], is_train)
    RELU_out1 = ReLu_layer(BN_out1)
    # The 1st conv layer
    with tf.variable_scope('conv1'):
        CONV_out1 = conv_layer(data_format, RELU_out1, 1, 'SAME', shape_conv1)
    # The 2nd bn layer
    BN_out2 = BN(data_format, CONV_out1, 'bn2', shape_conv2[2], is_train)
    RELU_out2 = ReLu_layer(BN_out2)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        CONV_out2 = conv_layer(data_format, RELU_out2, 1, 'SAME', shape_conv2)
    # Fuse
    RES_out = tf.add(input_tensor, CONV_out2)

    return RES_out


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


def get_conv_transpose_ksize(factor):
    # Input: specify upsampling factor
    return 2 * factor - factor % 2


def upsample_filt(size):
    """
    Make a 2D bilinear kernel suitable for upsampling of the given (h, w) kernel size.
    """
    factor = (size + 1) // 2
    if size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = np.ogrid[:size, :size]
    return (1 - abs(og[0] - center) / factor) * \
           (1 - abs(og[1] - center) / factor)


def set_conv_transpose_filters(variables):
    # Input: all tf.Variables
    set_filter_ops = []
    for v in variables:
        if '-up' in v.name:
            h, w, k, m = v.get_shape()
            tmp = np.zeros((m, k, h, w))
            if m != k:
                sys.exit("Transpose kernel input and output channels must be the same.")
            if h != w:
                sys.exit("Transpose kernel height and width must be the same. ")
            up_filter = upsample_filt(int(h))
            tmp[range(m), range(k), :, :] = up_filter
            set_filter_ops.append(tf.assign(v, tmp.transpose(2, 3, 1, 0)))

    return set_filter_ops


def conv_transpose(data_format, input_tensor, factor=0, padding='SAME'):
    # Create a transposed conv layer according to upsampling factor.
    # This function must be put into an appropriate variable scope.
    # All transposed filters must be reset to bilinear after tf.train.Saver store the variables

    batch_size = tf.shape(input_tensor)[0]
    trans_ksize = get_conv_transpose_ksize(factor)
    init_filter = tf.get_variable('bilinear-up',
                                  shape=[trans_ksize, trans_ksize, 16, 16],
                                  initializer = tf.zeros_initializer(),
                                  trainable = False)
    if data_format == "NCHW":
        out_channels = tf.shape(input_tensor)[1]
        new_H = tf.shape(input_tensor)[2]
        new_W = tf.shape(input_tensor)[3]
        trans_output_shape = [batch_size, out_channels, new_H, new_W]
        trans_stride = [1, 1, factor, factor]
    else:
        out_channels = tf.shape(input_tensor)[3]
        new_H = tf.shape(input_tensor)[1]
        new_W = tf.shape(input_tensor)[2]
        trans_output_shape = [batch_size, new_H, new_W, out_channels]
        trans_stride = [1, factor, factor, 1]

    trans_out = tf.nn.conv2d_transpose(input_tensor,
                                       init_filter,
                                       trans_output_shape,
                                       trans_stride,
                                       padding,
                                       data_format)

    return trans_out

def crop_features(data_format, feature, out_size):
    # Slice down the feature to out_size
    # This is done after deconv since deconv output may exceed original image size

    up_size = tf.shape(feature)
    if data_format == "NCHW":
        ini_h = tf.div(tf.subtract(up_size[2], out_size[2]), 2) # might be zero
        ini_w = tf.div(tf.subtract(up_size[3], out_size[3]), 2) # might be zero
        slice_input = tf.slice(feature, (0, 0, ini_h, ini_w), (-1, -1, out_size[2], out_size[3]))
        return tf.reshape(slice_input, [int(feature.get_shape()[0]), int(feature.get_shape()[1]), out_size[2], out_size[3]])
    else:
        ini_h = tf.div(tf.subtract(up_size[1], out_size[1]), 2)  # might be zero
        ini_w = tf.div(tf.subtract(up_size[2], out_size[2]), 2)  # might be zero
        slice_input = tf.slice(feature, (0, ini_h, ini_w, 0), (-1, out_size[1], out_size[2], -1))
        return tf.reshape(slice_input, [int(feature.get_shape()[0]), out_size[1], out_size[2], int(feature.get_shape()[3])])




