from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
import dataset as dt
# from core import resnet38

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.5

# config dataset params (3 datasets with different scales)
params_data50 = {
    'mode': 'parent_train_binary',
    'batch': 2,
    'tfrecord': '/work/wangyu/davis_train_50.tfrecord'
}
params_data80 = {
    'mode': 'parent_train_binary',
    'batch': 2,
    'tfrecord': '/work/wangyu/davis_train_80.tfrecord'
}
params_data100 = {
    'mode': 'parent_train_binary',
    'batch': 2,
    'tfrecord': '/work/wangyu/davis_train_100.tfrecord'
}
with tf.device('/cpu:0'):
    dataset_50 = dt(params_data50)
    dataset_80 = dt(params_data80)
    dataset_100 = dt(params_data100)
    iter_50 = dataset_50.get_iterator()
    iter_80 = dataset_80.get_iterator()
    iter_100 = dataset_100.get_iterator()
    handle = tf.placeholder(tf.string, shape=[])
    iterator = tf.contrib.data.Iterator.from_string_handle(handle, iter_50.output_types)
    next_batch = iterator.get_next()


# config train params
params_model = {
    'feed_weight': 'ImgNet',
    'batch': 2,
    'l2_weight': 1e-5,
    'lr': 5e-6,
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/',
    'tsboard_logs': '../data/tsboard_logs/'
}

# TODO: epochs, global_steps
dump_img = next_batch['img']
dump_gt = next_batch['gt']

sum_img = tf.summary.image('input_img', dump_img)
sum_gt = tf.summary.image('input_gt', dump_gt)
sum_all = tf.summary.merge_all()

# TODO: Build network, on GPU by default
init_op = tf.global_variables_initializer()

# run the session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.Filewriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)
    iter_50_handle = sess.run(iter_50.string_handle())
    iter_80_handle = sess.run(iter_80.string_handle())
    iter_100_handle = sess.run(iter_100.string_handle())

    img_, gt_, sum_all_ = sess.run([dump_img, dump_gt, sum_all], feed_dict={handle: iter_50_handle})
    sum_writer.add_summary(sum_all_, 0)
    img_, gt_, sum_all_ = sess.run([dump_img, dump_gt, sum_all], feed_dict={handle: iter_80_handle})
    sum_writer.add_summary(sum_all_, 1)
    img_, gt_, sum_all_ = sess.run([dump_img, dump_gt, sum_all], feed_dict={handle: iter_100_handle})
    sum_writer.add_summary(sum_all_, 2)





