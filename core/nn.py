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

    # The 1st bn layer
    bn_out1 = BN(data_format, input_tensor, 'bn1', shape_dict['convs'][0][2], is_train)
    relu_out1 = ReLu_layer(bn_out1)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['side'])
    # The 1st conv
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['convs'][0])
    # The 2nd bn layer
    bn_out2 = BN(data_format, conv_out1, 'bn2', shape_dict['convs'][1][2], is_train)
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
    bn_out1 = BN(data_format, input_tensor, 'bn1', shape_conv1[2], is_train)
    relu_out1 = ReLu_layer(bn_out1)
    # The 1st conv layer
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_conv1)
    # The 2nd bn layer
    bn_out2 = BN(data_format, conv_out1, 'bn2', shape_conv2[2], is_train)
    relu_out2 = ReLu_layer(bn_out2)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_conv2)
    # Fuse
    block_out = tf.add(input_tensor, conv_out2)

    return block_out


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


def BN(data_format, input_tensor, bn_scope=None, shape=None, is_train=False):

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


def conv_transpose(data_format, input_tensor, filter_size, factor=0, padding='SAME'):
    # Create a transposed conv layer according to upsampling factor.
    # This function must be put into an appropriate variable scope.
    # All transposed filters must be reset to bilinear after tf.train.Saver store the variables

    # filter_size
    in_channel = filter_size[0]
    out_channel = filter_size[1]

    batch_size = tf.shape(input_tensor)[0]
    trans_ksize = get_conv_transpose_ksize(factor)
    init_filter = tf.get_variable('bilinear-up',
                                  shape=[trans_ksize, trans_ksize, out_channel, in_channel],
                                  initializer = tf.zeros_initializer(),
                                  trainable = False)
    if data_format == "NCHW":
        # out_channels = tf.shape(input_tensor)[1]
        out_channels = out_channel
        new_H = tf.shape(input_tensor)[2] * factor
        new_W = tf.shape(input_tensor)[3] * factor
        trans_output_shape = [batch_size, out_channels, new_H, new_W]
        trans_stride = [1, 1, factor, factor]
    else:
        # out_channels = tf.shape(input_tensor)[3]
        out_channels = out_channel
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
        return tf.reshape(slice_input, [up_size[0], up_size[1], out_size[2], out_size[3]])
    else:
        ini_h = tf.div(tf.subtract(up_size[1], out_size[1]), 2)  # might be zero
        ini_w = tf.div(tf.subtract(up_size[2], out_size[2]), 2)  # might be zero
        slice_input = tf.slice(feature, (0, ini_h, ini_w, 0), (-1, out_size[1], out_size[2], -1))
        return tf.reshape(slice_input, [up_size[0], out_size[1], out_size[2], up_size[3]])

def bias_layer(data_format, input_tensor, shape=None):

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







