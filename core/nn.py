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

def create_conv_kernel(shape=None):
    '''
    :param shape: the shape of kernel to be created
    :return: a tf.tensor
    '''
    init_op = tf.truncated_normal_initializer(stddev=0.001)
    var = tf.get_variable(name='kernel', shape=shape, initializer=init_op)

    return var


