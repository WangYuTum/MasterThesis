'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import sys


def lstm_conv2d(data_format, input_tensor):
    '''
    :param data_format: 'NCHW' or 'NHWC'
    :param input_tensor: [2,128,30,56] or [2,30,56,128], first dim indicates max_time
    :return: out of lstm [2,128,30,56] or [2,30,56,128]
    '''

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    if data_format == "NCHW":
        input_tensor = tf.transpose(input_tensor, [0,2,3,1]) # To NHWC
    if tf.contrib.framework.get_name_scope().find('attention') != -1:
        ker_shape = [3, 3] # large kernel shape for attention branch
    else:
        ker_shape = [1, 1] # small kernel shape for seg branch
    input_tensor = tf.expand_dims(input_tensor, 0) # [1,2,30,56,128], [batch, time_max, h, w, 128]
    lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[30, 56, 128],
                                              output_channels=128,
                                              kernel_shape=ker_shape,
                                              use_bias=True,    # default
                                              skip_connection=False, # default
                                              forget_bias=1.0,  # default
                                              initializers=None,    # default
                                              name='conv2dlstm')
    zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
    #input_tensor = tf.reshape(input_tensor, [1,2,30,56,128])
    lstm_out, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                              inputs=input_tensor,
                                              sequence_length=[2],
                                              initial_state=zero_state,
                                              dtype=tf.float32,
                                              swap_memory=True)
    # lstm_out has shape: [1,2,30,56,128]
    lstm_out = tf.squeeze(lstm_out, 0)  # to [2,30,56,128]
    # change back to required data_format
    if data_format == "NCHW":
        lstm_out = tf.transpose(lstm_out, [0,3,1,2])

    return lstm_out


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


def res_side(data_format, input_tensor, shape_dict):
    ''' The residual block unit with side conv '''

    # The 1st activation
    relu_out1 = ReLu_layer(input_tensor)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['side'])
    # The 1st conv
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['convs'][0])
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_dict['convs'][1])
    # Fuse
    block_out = tf.add(side_out, conv_out2)

    return  block_out


def res(data_format, input_tensor, shape_dict):
    ''' The residual block unit with shortcut '''

    scope_name = tf.get_variable_scope().name
    if scope_name.find('B4') != -1:
        shape_conv1 = shape_dict[0]
        shape_conv2 = shape_dict[1]
    else:
        shape_conv1 = shape_dict[1]
        shape_conv2 = shape_dict[1]

    # The 1st activation
    relu_out1 = ReLu_layer(input_tensor)
    # The 1st conv layer
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_conv1)
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
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
    vars_lr['main/feat_reduce/kernel'] = 1.0
    vars_lr['main/feat_reduce/bias'] = 2.0

    # TODO
    vars_lr['segmentation/seg_lstm2d/...'] = 1.0
    vars_lr['segmentation/lstm2d_decode/kernel'] = 1.0
    vars_lr['segmentation/B1_side_path/kernel'] = 1.0
    vars_lr['segmentation/B1_side_path/bias'] = 2.0
    vars_lr['segmentation/B2_side_path/kernel'] = 1.0
    vars_lr['segmentation/B2_side_path/bias'] = 2.0
    vars_lr['segmentation/B3_side_path/kernel'] = 1.0
    vars_lr['segmentation/B3_side_path/bias'] = 2.0
    vars_lr['segmentation/B4_side_path/kernel'] = 1.0
    vars_lr['segmentation/B4_side_path/bias'] = 2.0
    vars_lr['segmentation/lstm_decoded/kernel'] = 1.0
    vars_lr['segmentation/lstm_decoded/bias'] = 2.0
    vars_lr['segmentation/fuse/kernel'] = 0.01
    vars_lr['segmentation/fuse/bias'] = 0.02

    vars_lr['attention/fuse/kernel'] = 1.0
    vars_lr['attention/fuse/bias'] = 2.0
    vars_lr['attention/reduce/kernel'] = 1.0
    vars_lr['attention/reduce/bias'] = 1.0
    # TODO
    vars_lr['attention/att_lstm/...'] = 1.0
    vars_lr['attention/lstm2d_decode/kernel'] = 1.0
    vars_lr['attention/up/kernel'] = 1.0
    vars_lr['attention/up/bias'] = 1.0

    return vars_lr









