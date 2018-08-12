'''
    Basic building blocks for resnet.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def lstm_conv2d_train(data_format, input_tensor):
    '''
    :param data_format: 'NCHW' or 'NHWC'
    :param input_tensor: [8,65,256,512] or [8,256,512,65]
    :return: out of lstm [8,65,256,512] or [8,256,512,65]
    '''

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    if data_format == "NCHW":
        input_tensor = tf.transpose(input_tensor, [0,2,3,1]) # To NHWC
    ker_shape = [3, 3]
    lstm_in = tf.expand_dims(input_tensor, 0) #  [1, 10, 256, 512, 65] [batch, max_time, h, w, c]
    # lstm_in = tf.stack([input_tensor[0:4,:,:,:], input_tensor[4:8,:,:,:]], axis=0) # [2, 4, 256, 512, 65] [batch, max_time, h, w, c]
    lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[256, 512, 65],
                                              output_channels=64,
                                              kernel_shape=ker_shape,
                                              use_bias=True,    # default
                                              skip_connection=False, # default
                                              forget_bias=1.0,  # default
                                              initializers=None,    # default
                                              name='conv2dlstm')
    zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
    # zero_state = lstm_cell.zero_state(batch_size=2, dtype=tf.float32)
    lstm_out, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                              inputs=lstm_in,
                                              sequence_length=[10], #[4,4],
                                              initial_state=zero_state,
                                              dtype=tf.float32,
                                              swap_memory=True)
    # lstm_out has shape: [1,10,256,512,65], must restore shape of [10, 256, 512, 64]
    lstm_out = tf.squeeze(lstm_out, 0)
    # # lstm_out has shape: [2,4,256,512,65], must restore shape of [8, 256, 512, 64]
    # lstm_out = tf.concat([tf.squeeze(lstm_out[0:1,:,:,:,:]), tf.squeeze(lstm_out[1:2,:,:,:,:])], axis=0)
    # change back to required data_format
    if data_format == "NCHW":
        lstm_out = tf.transpose(lstm_out, [0,3,1,2])

    return lstm_out

def lstm_conv2d_inf(data_format, input_tensor, feed_state):
    '''
    :param data_format: 'NCHW' or 'NHWC'
    :param input_tensor: [1,65,256,512] or [1,256,512,65], for batch=1, one_time_step
    :param feed_state: [2, 256, 512, 64], Must be converted to a tuple ([1,256,512,64], [1,256,512,64])
    :return: lstm_out: [1,64,256,512] or [1,256,512,64], lstm_state: lstm_state_tuple, assign_state_ops
    '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)
    if data_format == "NCHW":
        input_tensor = tf.transpose(input_tensor, [0,2,3,1]) # To NHWC
    ker_shape = [3, 3]
    input_tensor = tf.expand_dims(input_tensor, 0)  # [1,1,256,512,65], [batch, time_max, h, w, 65]
    lstm_cell = tf.contrib.rnn.Conv2DLSTMCell(input_shape=[256, 512, 65],
                                              output_channels=64,
                                              kernel_shape=ker_shape,
                                              use_bias=True,  # default
                                              skip_connection=False,  # default
                                              forget_bias=1.0,  # default
                                              initializers=None,  # default
                                              name='conv2dlstm')
    # zero_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)
    c_var = tf.Variable(tf.zeros([1,256,512,64]), trainable=False, dtype=tf.float32, name='LSTM2d_c')
    h_var = tf.Variable(tf.zeros([1,256,512,64]), trainable=False, dtype=tf.float32, name='LSTM2d_h')
    new_tuple = tf.contrib.rnn.LSTMStateTuple(c_var, h_var)
    assign_c_op = tf.assign(c_var, feed_state[0:1,:,:,:])
    assign_h_op = tf.assign(h_var, feed_state[1:2,:,:,:])
    assign_state_ops = [assign_c_op, assign_h_op]
    lstm_out, final_state = tf.nn.dynamic_rnn(cell=lstm_cell,
                                              inputs=input_tensor,
                                              sequence_length=[1],
                                              initial_state=new_tuple,
                                              dtype=tf.float32,
                                              swap_memory=True)
    # lstm_out has shape: [1,1,256,512,64], final_state: ([1,256,512,64], [1,256,512,64])
    lstm_out = tf.squeeze(lstm_out, 0)  # to [1,256,512,64]
    # change back to required data_format
    if data_format == "NCHW":
        lstm_out = tf.transpose(lstm_out, [0,3,1,2])

    return lstm_out, final_state, assign_state_ops


def conv_layer(data_format, input_tensor, stride=1, padding='SAME', shape=None, training=False):
    ''' The standard convolution layer '''
    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    trainable = False
    # if training, only train lstm branch
    if training:
        if scope_name.find('main') != -1:
            trainable = False
        else:
            trainable = True
    else:
            trainable = False

    kernel = create_conv_kernel(shape, trainable)

    if data_format == "NCHW":
        conv_stride = [1, 1, stride, stride]
    else:
        conv_stride = [1, stride, stride, 1]
    conv_out = tf.nn.conv2d(input_tensor, kernel, strides=conv_stride, padding=padding, data_format=data_format)

    return conv_out


def res_side(data_format, input_tensor, shape_dict, training=False):
    ''' The residual block unit with side conv '''

    # The 1st activation
    relu_out1 = ReLu_layer(input_tensor)

    # The side conv
    with tf.variable_scope('side'):
        side_out = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['side'], training)
    # The 1st conv
    with tf.variable_scope('conv1'):
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_dict['convs'][0], training)
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_dict['convs'][1], training)
    # Fuse
    block_out = tf.add(side_out, conv_out2)

    return  block_out


def res(data_format, input_tensor, shape_dict, training):
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
        conv_out1 = conv_layer(data_format, relu_out1, 1, 'SAME', shape_conv1, training)
    # The 2nd activation
    relu_out2 = ReLu_layer(conv_out1)
    # The 2nd conv layer
    with tf.variable_scope('conv2'):
        conv_out2 = conv_layer(data_format, relu_out2, 1, 'SAME', shape_conv2, training)
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


def bias_layer(data_format, input_tensor, shape=None, training=False):

    scope_name = tf.get_variable_scope().name
    print('Layer name: %s'%scope_name)

    trainable = False
    # if training, only train lstm branch
    if training:
        if scope_name.find('main') != -1:
            trainable = False
        else:
            trainable = True
    else:
            trainable = False

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

def get_parent_var():

    ## The main cnn part weights
    cnn_dict = {}
    # for the first conv
    with tf.variable_scope('main/B0', reuse=True):
        cnn_dict['main/B0/kernel'] = tf.get_variable('kernel')
    # for all resnet side convs
    for i in range(4):
        with tf.variable_scope('main/B' + str(i + 1) + '_0/side', reuse=True):
            cnn_dict['main/B' + str(i + 1) + '_0/side/kernel'] = tf.get_variable('kernel')
    # for all convs on the main path
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv1', reuse=True):
                cnn_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel'] = tf.get_variable('kernel')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv2', reuse=True):
                cnn_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel'] = tf.get_variable('kernel')
    # for convs on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv1', reuse=True):
            cnn_dict['main/B3_' + str(i + 3) + '/conv1/kernel'] = tf.get_variable('kernel')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv2', reuse=True):
            cnn_dict['main/B3_' + str(i + 3) + '/conv2/kernel'] = tf.get_variable('kernel')

    # for the side path
    for i in range(4):
        with tf.variable_scope('main/B' + str(i+1) + '_side_path', reuse=True):
            cnn_dict['main/B' + str(i + 1) + '_side_path/kernel'] = tf.get_variable('kernel')
            cnn_dict['main/B' + str(i + 1) + '_side_path/bias'] = tf.get_variable('bias')

    return cnn_dict

def get_main_var():

    ## The main cnn part weights
    cnn_dict = {}
    # for the first conv
    with tf.variable_scope('main/B0', reuse=True):
        cnn_dict['main/B0/kernel'] = tf.get_variable('kernel')
    # for all resnet side convs
    for i in range(4):
        with tf.variable_scope('main/B' + str(i + 1) + '_0/side', reuse=True):
            cnn_dict['main/B' + str(i + 1) + '_0/side/kernel'] = tf.get_variable('kernel')
    # for all convs on the main path
    for i in range(4):
        for j in range(3):
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv1', reuse=True):
                cnn_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv1/kernel'] = tf.get_variable('kernel')
            with tf.variable_scope('main/B' + str(i + 1) + '_' + str(j) + '/conv2', reuse=True):
                cnn_dict['main/B' + str(i + 1) + '_' + str(j) + '/conv2/kernel'] = tf.get_variable('kernel')
    # for convs on B3_3, B3_4, B3_5
    for i in range(3):
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv1', reuse=True):
            cnn_dict['main/B3_' + str(i + 3) + '/conv1/kernel'] = tf.get_variable('kernel')
        with tf.variable_scope('main/B3_' + str(i + 3) + '/conv2', reuse=True):
            cnn_dict['main/B3_' + str(i + 3) + '/conv2/kernel'] = tf.get_variable('kernel')

    # for the side path
    for i in range(4):
        with tf.variable_scope('main/B' + str(i+1) + '_side_path', reuse=True):
            cnn_dict['main/B' + str(i + 1) + '_side_path/kernel'] = tf.get_variable('kernel')
            cnn_dict['main/B' + str(i + 1) + '_side_path/bias'] = tf.get_variable('bias')

    # for the fuse conv
    with tf.variable_scope('main/fuse', reuse=True):
        cnn_dict['main/fuse/kernel'] = tf.get_variable('kernel')
        cnn_dict['main/fuse/bias'] = tf.get_variable('bias')

    return cnn_dict

def get_lstm_var():

    lstm_dict = {}
    # for conv before lstm
    with tf.variable_scope('feat_update/conv_in_lstm', reuse=True):
        lstm_dict['feat_update/conv_in_lstm/bias'] = tf.get_variable('bias')
        lstm_dict['feat_update/conv_in_lstm/kernel'] = tf.get_variable('kernel')

    # for lstm weights
    with tf.variable_scope('feat_update/lstm_2d/rnn/conv2dlstm', reuse=True):
        lstm_dict['feat_update/lstm_2d/rnn/conv2dlstm/biases'] = tf.get_variable('biases')
        lstm_dict['feat_update/lstm_2d/rnn/conv2dlstm/kernel'] = tf.get_variable('kernel')

    # for conv after lstm
    with tf.variable_scope('feat_update/conv_out_lstm', reuse=True):
        lstm_dict['feat_update/conv_out_lstm/bias'] = tf.get_variable('bias')
        lstm_dict['feat_update/conv_out_lstm/kernel'] = tf.get_variable('kernel')

    # for the cls layer
    with tf.variable_scope('feat_update/conv_cls', reuse=True):
        lstm_dict['feat_update/conv_cls/bias'] = tf.get_variable('bias')
        lstm_dict['feat_update/conv_cls/kernel'] = tf.get_variable('kernel')

    return lstm_dict

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

    vars_lr['main/fuse/kernel'] = 0.01
    vars_lr['main/fuse/bias'] = 0.02

    return vars_lr









