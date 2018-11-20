'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import sys
import numpy as np


def conv_layer(data_format, input_tensor, stride=1, padding='SAME', shape=None, train_flag=None):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    trainable = False
    if train_flag == 0:
        if scope_name.find('main') != -1 or scope_name.find('classifier') != -1:
            trainable = True
        else:
            trainable = False
    elif train_flag == 1:
        if scope_name.find('feat_transform') != -1 or scope_name.find('classifier') != -1:
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

def max_pool2d_4(data_format, input_tensor, stride=4, padding='SAME'):
    ''' The standard max_pool2d with kernel size 4x4 '''

    if data_format == "NCHW":
        pool_size = [1, 1, 4, 4]
        pool_stride = [1, 1, stride, stride]
    else:
        pool_size = [1, 4, 4, 1]
        pool_stride = [1, 4, 4, 1]

    out = tf.nn.max_pool(input_tensor, pool_size, pool_stride, padding, data_format)

    return out

def avg_pool2d(data_format, input_tensor, size=2, stride=2, padding='SAME'):
    ''' The standard avg_pool2d '''

    if data_format == "NCHW":
        pool_size = [1, 1, size, size]
        pool_stride = [1, 1, stride, stride]
    else:
        pool_size = [1, size, size, 1]
        pool_stride = [1, stride, stride, 1]

    out = tf.nn.avg_pool(input_tensor, pool_size, pool_stride, padding, data_format)

    return out

def ReLu_layer(input_tensor):

    relu_out = tf.nn.relu(input_tensor)

    return relu_out


def bias_layer(data_format, input_tensor, shape=None, train_flag=None):

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    trainable = False
    if train_flag == 0:
        if scope_name.find('main') != -1 or scope_name.find('classifier') != -1:
            trainable = True
        else:
            trainable = False
    elif train_flag == 1:
        if scope_name.find('feat_transform') != -1 or scope_name.find('classifier') != -1:
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
    with tf.variable_scope('main/B0', reuse=tf.AUTO_REUSE):
        imgnet_dict['main/B0/kernel'] = tf.get_variable('kernel')
    # for all resnet side convs
    for i in range(4):
        with tf.variable_scope('main/B' + str(i + 1) + '_0/side', reuse=tf.AUTO_REUSE):
            imgnet_dict['main/B' + str(i + 1) + '_0/side/kernel'] = tf.get_variable('kernel')
    # for all convs on the main path
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv1', reuse=tf.AUTO_REUSE):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel'] = tf.get_variable('kernel')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv2', reuse=tf.AUTO_REUSE):
                imgnet_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel'] = tf.get_variable('kernel')
    # for convs on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv1', reuse=tf.AUTO_REUSE):
            imgnet_dict['main/B3_' + str(i + 3) + '/conv1/kernel'] = tf.get_variable('kernel')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv2', reuse=tf.AUTO_REUSE):
            imgnet_dict['main/B3_' + str(i + 3) + '/conv2/kernel'] = tf.get_variable('kernel')

    # for side_path 
    for i in range(4):
        with tf.variable_scope('main/B' + str(i+1) + '_side_path', reuse=tf.AUTO_REUSE):
            imgnet_dict['main/B' + str(i+1) + '_side_path/kernel'] = tf.get_variable('kernel')
            imgnet_dict['main/B' + str(i+1) + '_side_path/bias'] = tf.get_variable('bias')

    # for main fuse
    with tf.variable_scope('main/fuse', reuse=tf.AUTO_REUSE):
        imgnet_dict['main/fuse/kernel'] = tf.get_variable('kernel')
        imgnet_dict['main/fuse/bias'] = tf.get_variable('bias')

    # for global_step
    # with tf.variable_scope('', reuse=True):
    #     imgnet_dict['global_step'] = tf.get_variable('global_step')

    return imgnet_dict

def get_full_var():

    ## Full variables auto-reuse if some Adam-opt vars are missing
    full_dict = {}
    with tf.variable_scope('main/B0', reuse=True):
        full_dict['main/B0/kernel'] = tf.get_variable('kernel')
        full_dict['main/B0/kernel/Adam'] = tf.get_variable('kernel/Adam')
        full_dict['main/B0/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')

    # for all resnet side convs
    for i in range(4):
        with tf.variable_scope('main/B' + str(i + 1) + '_0/side', reuse=True):
            full_dict['main/B' + str(i + 1) + '_0/side/kernel'] = tf.get_variable('kernel')
            full_dict['main/B' + str(i + 1) + '_0/side/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['main/B' + str(i + 1) + '_0/side/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
    # for all convs on the main path
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv1', reuse=True):
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel'] = tf.get_variable('kernel')
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel/Adam'] = tf.get_variable('kernel/Adam')
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv2', reuse=True):
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel'] = tf.get_variable('kernel')
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel/Adam'] = tf.get_variable('kernel/Adam')
                full_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
    # for convs on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv1', reuse=True):
            full_dict['main/B3_' + str(i + 3) + '/conv1/kernel'] = tf.get_variable('kernel')
            full_dict['main/B3_' + str(i + 3) + '/conv1/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['main/B3_' + str(i + 3) + '/conv1/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv2', reuse=True):
            full_dict['main/B3_' + str(i + 3) + '/conv2/kernel'] = tf.get_variable('kernel')
            full_dict['main/B3_' + str(i + 3) + '/conv2/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['main/B3_' + str(i + 3) + '/conv2/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')

    # for side_path
    for i in range(4):
        with tf.variable_scope('main/B' + str(i+1) + '_side_path', reuse=True):
            full_dict['main/B' + str(i+1) + '_side_path/kernel'] = tf.get_variable('kernel')
            full_dict['main/B' + str(i + 1) + '_side_path/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['main/B' + str(i + 1) + '_side_path/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
            full_dict['main/B' + str(i+1) + '_side_path/bias'] = tf.get_variable('bias')
            full_dict['main/B' + str(i + 1) + '_side_path/bias/Adam'] = tf.get_variable('bias/Adam')
            full_dict['main/B' + str(i + 1) + '_side_path/bias/Adam_1'] = tf.get_variable('bias/Adam_1')

    # for main fuse
    with tf.variable_scope('main/fuse', reuse=True):
        full_dict['main/fuse/kernel'] = tf.get_variable('kernel')
        full_dict['main/fuse/kernel/Adam'] = tf.get_variable('kernel/Adam')
        full_dict['main/fuse/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
        full_dict['main/fuse/bias'] = tf.get_variable('bias')
        full_dict['main/fuse/bias/Adam'] = tf.get_variable('bias/Adam')
        full_dict['main/fuse/bias/Adam_1'] = tf.get_variable('bias/Adam_1')

    # for optical_flow
    for i in range(5):
        with tf.variable_scope('optical_flow/B'+str(i)+'/conv1', reuse=True):
            full_dict['optical_flow/B'+str(i)+'/conv1/kernel'] = tf.get_variable('kernel')
            full_dict['optical_flow/B' + str(i) + '/conv1/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['optical_flow/B' + str(i) + '/conv1/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
        with tf.variable_scope('optical_flow/B'+str(i)+'/conv2', reuse=True):
            full_dict['optical_flow/B' + str(i) + '/conv2/kernel'] = tf.get_variable('kernel')
            full_dict['optical_flow/B' + str(i) + '/conv2/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['optical_flow/B' + str(i) + '/conv2/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')

    # for feat_transform convs
    for i in range(5):
        if i != 4:
            num_convs = i + 1
        else:
            num_convs = 4
        for conv_id in range(num_convs):
            with tf.variable_scope('feat_transform/B'+str(i)+'/conv'+str(conv_id+1), reuse=True):
                full_dict['feat_transform/B'+str(i)+'/conv'+str(conv_id+1)+'/kernel'] = tf.get_variable('kernel')
                full_dict['feat_transform/B'+str(i)+'/conv'+str(conv_id+1)+'/kernel/Adam'] = tf.get_variable('kernel/Adam')
                full_dict['feat_transform/B'+str(i)+'/conv'+str(conv_id+1)+'/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')

    # for feat_transform side paths
    for i in range(5):
        with tf.variable_scope('feat_transform/B'+str(i)+'_trans_side_path', reuse=True):
            full_dict['feat_transform/B'+str(i)+'_trans_side_path/kernel'] = tf.get_variable('kernel')
            full_dict['feat_transform/B' + str(i) + '_trans_side_path/kernel/Adam'] = tf.get_variable('kernel/Adam')
            full_dict['feat_transform/B' + str(i) + '_trans_side_path/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
            full_dict['feat_transform/B'+str(i)+'_trans_side_path/bias'] = tf.get_variable('bias')
            full_dict['feat_transform/B' + str(i) + '_trans_side_path/bias/Adam'] = tf.get_variable('bias/Adam')
            full_dict['feat_transform/B' + str(i) + '_trans_side_path/bias/Adam_1'] = tf.get_variable('bias/Adam_1')

    # for feat_transform fuse
    with tf.variable_scope('feat_transform/fuse', reuse=True):
        full_dict['feat_transform/fuse/kernel'] = tf.get_variable('kernel')
        full_dict['feat_transform/fuse/kernel/Adam'] = tf.get_variable('kernel/Adam')
        full_dict['feat_transform/fuse/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
        full_dict['feat_transform/fuse/bias'] = tf.get_variable('bias')
        full_dict['feat_transform/fuse/bias/Adam'] = tf.get_variable('bias/Adam')
        full_dict['feat_transform/fuse/bias/Adam_1'] = tf.get_variable('bias/Adam_1')

    # for final classifier
    with tf.variable_scope('classifier', reuse=True):
        full_dict['classifier/kernel'] = tf.get_variable('kernel')
        full_dict['classifier/kernel/Adam'] = tf.get_variable('kernel/Adam')
        full_dict['classifier/kernel/Adam_1'] = tf.get_variable('kernel/Adam_1')
        full_dict['classifier/bias'] = tf.get_variable('bias')
        full_dict['classifier/bias/Adam'] = tf.get_variable('bias/Adam')
        full_dict['classifier/bias/Adam_1'] = tf.get_variable('bias/Adam_1')

    # for global_step
    with tf.variable_scope('', reuse=True):
        full_dict['global_step'] = tf.get_variable('global_step')


    return full_dict

def get_main_Adam(reader_main):

    all_adams = []
    B0 = {}
    tmp_adam = get_Adam(reader_main, 'main/B0/kernel/Adam')
    tmp_adam1 = get_Adam(reader_main, 'main/B0/kernel/Adam_1')
    B0['adam'] = tf.Variable(tmp_adam, name='main/B0/kernel/Adam')
    B0['adam1'] = tf.Variable(tmp_adam1, name='main/B0/kernel/Adam_1')
    all_adams.append(B0)

    B1_dict = {}
    for i in range(3):
        for conv_id in range(2):
            adam_name = 'main/B1_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam'
            adam1_name = 'main/B1_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam_1'
            tmp_adam = get_Adam(reader_main, adam_name)
            tmp_adam1 = get_Adam(reader_main, adam1_name)
            B1_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            B1_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(B1_dict)

    B2_dict = {}
    for i in range(3):
        for conv_id in range(2):
            adam_name = 'main/B2_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam'
            adam1_name = 'main/B2_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam_1'
            tmp_adam = get_Adam(reader_main, adam_name)
            tmp_adam1 = get_Adam(reader_main, adam1_name)
            B2_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            B2_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(B2_dict)

    B3_dict = {}
    for i in range(6):
        for conv_id in range(2):
            adam_name = 'main/B3_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam'
            adam1_name = 'main/B3_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam_1'
            tmp_adam = get_Adam(reader_main, adam_name)
            tmp_adam1 = get_Adam(reader_main, adam1_name)
            B3_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            B3_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(B3_dict)

    B4_dict = {}
    for i in range(3):
        for conv_id in range(2):
            adam_name = 'main/B4_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam'
            adam1_name = 'main/B4_' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam_1'
            tmp_adam = get_Adam(reader_main, adam_name)
            tmp_adam1 = get_Adam(reader_main, adam1_name)
            B4_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            B4_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(B4_dict)

    fuse_f_adam = get_Adam(reader_main, 'main/fuse/kernel/Adam')
    fuse_f_adam1 = get_Adam(reader_main, 'main/fuse/kernel/Adam_1')
    fuse_adam_f_var = tf.Variable(fuse_f_adam, name='main/fuse/kernel/Adam')
    fuse_adam1_f_var = tf.Variable(fuse_f_adam1, name='main/fuse/kernel/Adam_1')
    all_adams.append(fuse_adam_f_var)
    all_adams.append(fuse_adam1_f_var)

    fuse_b_adam = get_Adam(reader_main, 'main/fuse/bias/Adam')
    fuse_b_adam1 = get_Adam(reader_main, 'main/fuse/bias/Adam_1')
    fuse_adam_b_var = tf.Variable(fuse_b_adam, name='main/fuse/bias/Adam')
    fuse_adam1_b_var = tf.Variable(fuse_b_adam1, name='main/fuse/bias/Adam_1')
    all_adams.append(fuse_adam_b_var)
    all_adams.append(fuse_adam1_b_var)

    B_sides = {}
    for i in range(4):
        adam_name = 'main/B' + str(i + 1) + '_0/side/kernel/Adam'
        adam1_name = 'main/B' + str(i + 1) + '_0/side/kernel/Adam_1'
        tmp_adam = get_Adam(reader_main, adam_name)
        tmp_adam1 = get_Adam(reader_main, adam1_name)
        B_sides[adam_name] = tf.Variable(tmp_adam, name=adam_name)
        B_sides[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(B_sides)

    B_side_paths = {}
    for i in range(4):
        adam_kernel_name = 'main/B' + str(i + 1) + '_side_path/kernel/Adam'
        adam1_kernel_name = 'main/B' + str(i + 1) + '_side_path/kernel/Adam_1'
        adam_bias_name = 'main/B' + str(i + 1) + '_side_path/bias/Adam'
        adam1_bias_name = 'main/B' + str(i + 1) + '_side_path/bias/Adam_1'
        tmp_kernel_adam = get_Adam(reader_main, adam_kernel_name)
        tmp_kernel_adam1 = get_Adam(reader_main, adam1_kernel_name)
        tmp_b_adam = get_Adam(reader_main, adam_bias_name)
        tmp_b_adam1 = get_Adam(reader_main, adam1_bias_name)
        B_side_paths[adam_kernel_name] = tf.Variable(tmp_kernel_adam, name=adam_kernel_name)
        B_side_paths[adam1_kernel_name] = tf.Variable(tmp_kernel_adam1, name=adam1_kernel_name)
        B_side_paths[adam_bias_name] = tf.Variable(tmp_b_adam, name=adam_bias_name)
        B_side_paths[adam1_bias_name] = tf.Variable(tmp_b_adam1, name=adam1_bias_name)
    all_adams.append(B_side_paths)

    return all_adams

def get_OF_Feat_Adam(reader_OF):

    all_adams = []
    OF_dict = {}
    for i in range(5):
        for conv_id in range(2):
            adam_name = 'optical_flow/B' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam'
            adam1_name = 'optical_flow/B' + str(i) + '/conv' + str(conv_id + 1) + '/kernel/Adam_1'
            tmp_adam = get_Adam(reader_OF, adam_name)
            tmp_adam1 = get_Adam(reader_OF, adam1_name)
            OF_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            OF_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(OF_dict)

    Feat_dict = {}
    for i in range(5):
        if i != 4:
            num_convs = i + 1
        else:
            num_convs = 4
        for conv_id in range(num_convs):
            adam_name = 'feat_transform/B'+str(i)+'/conv'+str(conv_id+1)+'/kernel/Adam'
            adam1_name = 'feat_transform/B'+str(i)+'/conv'+str(conv_id+1)+'/kernel/Adam_1'
            tmp_adam = get_Adam(reader_OF, adam_name)
            tmp_adam1 = get_Adam(reader_OF, adam1_name)
            Feat_dict[adam_name] = tf.Variable(tmp_adam, name=adam_name)
            Feat_dict[adam1_name] = tf.Variable(tmp_adam1, name=adam1_name)
    all_adams.append(Feat_dict)

    Feat_sides_dict = {}
    for i in range(5):
        adam_kernel_name = 'feat_transform/B' + str(i) + '_trans_side_path/kernel/Adam'
        adam1_kernel_name = 'feat_transform/B' + str(i) + '_trans_side_path/kernel/Adam_1'
        adam_bias_name = 'feat_transform/B' + str(i) + '_trans_side_path/bias/Adam'
        adam1_bias_name = 'feat_transform/B' + str(i) + '_trans_side_path/bias/Adam_1'
        tmp_kernel_adam = get_Adam(reader_OF, adam_kernel_name)
        tmp_kernel_adam1 = get_Adam(reader_OF, adam1_kernel_name)
        tmp_b_adam = get_Adam(reader_OF, adam_bias_name)
        tmp_b_adam1 = get_Adam(reader_OF, adam1_bias_name)
        Feat_sides_dict[adam_kernel_name] = tf.Variable(tmp_kernel_adam, name=adam_kernel_name)
        Feat_sides_dict[adam1_kernel_name] = tf.Variable(tmp_kernel_adam1, name=adam1_kernel_name)
        Feat_sides_dict[adam_bias_name] = tf.Variable(tmp_b_adam, name=adam_bias_name)
        Feat_sides_dict[adam1_bias_name] = tf.Variable(tmp_b_adam1, name=adam1_bias_name)
    all_adams.append(Feat_sides_dict)

    feat_fuse_f_adam = get_Adam(reader_OF, 'feat_transform/fuse/kernel/Adam')
    feat_fuse_f_adam1 = get_Adam(reader_OF, 'feat_transform/fuse/kernel/Adam_1')
    feat_fuse_adam_f_var = tf.Variable(feat_fuse_f_adam, name='feat_transform/fuse/kernel/Adam')
    feat_fuse_adam1_f_var = tf.Variable(feat_fuse_f_adam1, name='feat_transform/fuse/kernel/Adam_1')
    all_adams.append(feat_fuse_adam_f_var)
    all_adams.append(feat_fuse_adam1_f_var)

    feat_fuse_b_adam = get_Adam(reader_OF, 'feat_transform/fuse/bias/Adam')
    feat_fuse_b_adam1 = get_Adam(reader_OF, 'feat_transform/fuse/bias/Adam_1')
    feat_fuse_adam_b_var = tf.Variable(feat_fuse_b_adam, name='feat_transform/fuse/bias/Adam')
    feat_fuse_adam1_b_var = tf.Variable(feat_fuse_b_adam1, name='feat_transform/fuse/bias/Adam_1')
    all_adams.append(feat_fuse_adam_b_var)
    all_adams.append(feat_fuse_adam1_b_var)

    return all_adams

def get_Adam(reader, tensor_key):
    ## Given reader, tensor_key/name
    ## Return np.float32 filters

    params = np.array(reader.get_tensor(tensor_key))
    params_shape = np.shape(params)
    return params

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

    vars_lr['feat_transform/B0/conv_resize/kernel'] = 1.0
    vars_lr['feat_transform/B1/conv_resize/kernel'] = 1.0
    vars_lr['feat_transform/B2/conv_resize/kernel'] = 1.0
    vars_lr['feat_transform/B3/conv_resize/kernel'] = 1.0
    vars_lr['feat_transform/B4/conv_resize/kernel'] = 1.0

    vars_lr['feat_transform/fuse/conv1/kernel'] = 1.0
    vars_lr['feat_transform/fuse/conv1/bias'] = 2.0
    vars_lr['feat_transform/fuse/conv2/kernel'] = 1.0
    vars_lr['feat_transform/fuse/conv2/bias'] = 2.0
    vars_lr['feat_transform/fuse/conv3/kernel'] = 1.0
    vars_lr['feat_transform/fuse/conv3/bias'] = 2.0
    vars_lr['feat_transform/fuse/fuse_trans/kernel'] = 1.0
    vars_lr['feat_transform/fuse/fuse_trans/bias'] = 2.0

    vars_lr['classifier/kernel'] = 0.1
    vars_lr['classifier/bias'] = 0.2

    return vars_lr