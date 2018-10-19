'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys


def conv_layer(data_format, input_tensor, stride=1, padding='SAME', shape=None, train_flag=None):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    trainable = False
    if train_flag == 0:
        if scope_name.find('main') != 0 or scope_name.find('classifier') != 0:
            trainable = True
        else:
            trainable = False
    elif train_flag == 1:
        if scope_name.find('optical_flow') != 0 or scope_name.find('feat_transform') != 0 or scope_name.find('classifier') != 0:
            trainable = True
        else:
            trainable = False
    elif train_flag == 2:
        trainable = True
    elif trainable == 3:
        trainable = False
    else:
        sys.exit('Non-valid train_flag.')

    kernel = create_conv_kernel(shape, trainable)

    if data_format == "NCHW":
        conv_stride = [1, 1, stride, stride]
    else:
        conv_stride = [1, stride, stride, 1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)

    return conv_out


def res_side(data_format, input_tensor, shape_dict, train_flag=None):
    ''' The residual block unit with side conv '''

    # The 1st activation
    relu_out1 = ReLu_layer(input_tensor)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['side'], train_flag)
    # The 1st conv
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['convs'][0], train_flag)
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_dict['convs'][1], train_flag)
    # Fuse
    block_out = tf.add(side_out, conv_out2)

    return  block_out


def res(data_format, input_tensor, shape_dict, train_flag=None):
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
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_conv1, train_flag)
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_conv2, train_flag)
    # Fuse
    block_out = tf.add(input_tensor, conv_out2)

    return block_out


def create_conv_kernel(shape=None, trainable=False):
    '''
    :param shape: the shape of kernel to be created
    :return: a tf.tensor
    '''
    init_op = tf.truncated_normal_initializer(stddev=0.001)
    var = tf.get_variable(name='kernel', shape=shape, initializer=init_op, trainable=trainable)

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


def bias_layer(data_format, input_tensor, shape=None, train_flag=None):

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    trainable = False
    if train_flag == 0:
        if scope_name.find('main') != 0 or scope_name.find('classifier') != 0:
            trainable = True
        else:
            trainable = False
    elif train_flag == 1:
        if scope_name.find('optical_flow') != 0 or scope_name.find('feat_transform') != 0 or scope_name.find('classifier') != 0:
            trainable = True
        else:
            trainable = False
    elif train_flag == 2:
        trainable = True
    elif train_flag == 3:
        trainable = False
    else:
        sys.exit('Non-valid train_flag.')

    bias = create_bias(shape, trainable)
    bias_out = tf.nn.bias_add(input_tensor, bias, data_format)

    return bias_out


def create_bias(shape=None, trainable=False):

    init = tf.zeros_initializer()
    var = tf.get_variable('bias', initializer=init, shape=shape, trainable=trainable)

    return var


def get_imgnet_var():

    ## Imgnet weights variables
    imgnet_dict = {}
    # for the first conv, dont use since input is now 3(rgb)+2(OF) channels
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

    # for side_path 
    for i in range(4):
        with tf.variable_scope('main/B' + str(i+1) + '_side_path', reuse=True):
            imgnet_dict['main/B' + str(i+1) + '_side_path/kernel'] = tf.get_variable('kernel')
            imgnet_dict['main/B' + str(i+1) + '_side_path/bias'] = tf.get_variable('bias')


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

    vars_lr['main/B1_side_path/kernel'] = 1.0
    vars_lr['main/B1_side_path/bias'] = 2.0
    vars_lr['main/B2_side_path/kernel'] = 1.0
    vars_lr['main/B2_side_path/bias'] = 2.0
    vars_lr['main/B3_side_path/kernel'] = 1.0
    vars_lr['main/B3_side_path/bias'] = 2.0
    vars_lr['main/B4_side_path/kernel'] = 1.0
    vars_lr['main/B4_side_path/bias'] = 2.0

    vars_lr['main/fuse/kernel'] = 1.0
    vars_lr['main/fuse/bias'] = 2.0

    vars_lr['optical_flow/B0/conv1/kernel'] = 1.0
    vars_lr['optical_flow/B0/conv2/kernel'] = 1.0
    vars_lr['optical_flow/B1/conv1/kernel'] = 1.0
    vars_lr['optical_flow/B1/conv2/kernel'] = 1.0
    vars_lr['optical_flow/B2/conv1/kernel'] = 1.0
    vars_lr['optical_flow/B2/conv2/kernel'] = 1.0
    vars_lr['optical_flow/B3/conv1/kernel'] = 1.0
    vars_lr['optical_flow/B3/conv2/kernel'] = 1.0
    vars_lr['optical_flow/B4/conv1/kernel'] = 1.0
    vars_lr['optical_flow/B4/conv2/kernel'] = 1.0
    vars_lr['feat_transform/B0/conv1/kernel'] = 1.0
    vars_lr['feat_transform/B1/conv1/kernel'] = 1.0
    vars_lr['feat_transform/B1/conv2/kernel'] = 1.0
    vars_lr['feat_transform/B2/conv1/kernel'] = 1.0
    vars_lr['feat_transform/B2/conv2/kernel'] = 1.0
    vars_lr['feat_transform/B2/conv3/kernel'] = 1.0
    vars_lr['feat_transform/B3/conv1/kernel'] = 1.0
    vars_lr['feat_transform/B3/conv2/kernel'] = 1.0
    vars_lr['feat_transform/B3/conv3/kernel'] = 1.0
    vars_lr['feat_transform/B3/conv4/kernel'] = 1.0
    vars_lr['feat_transform/B4/conv1/kernel'] = 1.0
    vars_lr['feat_transform/B4/conv2/kernel'] = 1.0
    vars_lr['feat_transform/B4/conv3/kernel'] = 1.0
    vars_lr['feat_transform/B4/conv4/kernel'] = 1.0

    vars_lr['feat_transform/B0_trans_side_path/kernel'] = 1.0
    vars_lr['feat_transform/B0_trans_side_path/bias'] = 2.0
    vars_lr['feat_transform/B1_trans_side_path/kernel'] = 1.0
    vars_lr['feat_transform/B1_trans_side_path/bias'] = 2.0
    vars_lr['feat_transform/B2_trans_side_path/kernel'] = 1.0
    vars_lr['feat_transform/B2_trans_side_path/bias'] = 2.0
    vars_lr['feat_transform/B3_trans_side_path/kernel'] = 1.0
    vars_lr['feat_transform/B3_trans_side_path/bias'] = 2.0
    vars_lr['feat_transform/B4_trans_side_path/kernel'] = 1.0
    vars_lr['feat_transform/B4_trans_side_path/bias'] = 2.0
    vars_lr['feat_transform/fuse/kernel'] = 1.0
    vars_lr['feat_transform/fuse/bias'] = 2.0

    vars_lr['classifier/kernel'] = 0.1
    vars_lr['classifier/bias'] = 0.2

    return vars_lr









