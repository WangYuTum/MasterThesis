from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
# from core import resnet38

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = ''
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
    dataset_50 = DAVIS_dataset(params_data50)
    dataset_80 = DAVIS_dataset(params_data80)
    dataset_100 = DAVIS_dataset(params_data100)
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
sum_gt = tf.summary.image('input_gt', tf.cast(dump_gt,tf.float16))
sum_all = tf.summary.merge_all()

# TODO: Build network, on GPU by default
init_op = tf.global_variables_initializer()

# run the session
# with tf.Session(config=config_gpu) as sess:
with tf.Session() as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)
    iter_50_handle = sess.run(iter_50.string_handle())
    iter_80_handle = sess.run(iter_80.string_handle())
    iter_100_handle = sess.run(iter_100.string_handle())

    # each step choose a random scale/dataset for training
    for step in range(3):
        rand_v = np.random.rand()
        train_scale = 0
        if rand_v <= 0.33:
            train_scale = 0
        elif rand_v <= 0.66:
            train_scale = 1
        else:
            train_scale = 2

        if train_scale == 0:
            feed_dict_v = {handle: iter_50_handle}
        elif train_scale == 1:
            feed_dict_v = {handle: iter_80_handle}
        else:
            feed_dict_v = {handle: iter_100_handle}
        img_, gt_, sum_all_ = sess.run([dump_img, dump_gt, sum_all], feed_dict=feed_dict_v)
        sum_writer.add_summary(sum_all_, step)





