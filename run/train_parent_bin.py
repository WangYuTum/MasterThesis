from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
from core import resnet
from core.nn import set_conv_transpose_filters
from core.nn import get_imgnet_var

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.95

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
    'batch': 2,
    'l2_weight': 0.0002,
    'init_lr': 1e-5, # original paper: 1e-8,
    'base_decay': 1.0,
    'sup_decay': 0.1,
    'fuse_decay': 0.01,
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/parent-sup/parent_binary_train.ckpt',
    'tsboard_logs': '../data/tsboard_logs/parent-sup/',
    'restore_imgnet': '../data/ckpts/imgnet.ckpt', # restore model from where
    'restore_parent_bin': '../data/ckpts/parent_binary_train.ckpt-xxx'
}
# define epochs
epochs = 5
num_frames = 4209 * 3
steps_per_epochs = int(num_frames / params_model['batch'])
global_step = 0
save_ckpt_interval = 6000
summary_write_interval = 50 # 50
print_screen_interval = 20

# display traning images/gts
feed_img = next_batch['img']
feed_gt = next_batch['gt']
feed_weight = next_batch['balanced_mat']
sum_img = tf.summary.image('input_img', feed_img)
sum_gt = tf.summary.image('input_gt', tf.cast(feed_gt,tf.float16))
sum_weight = tf.summary.image('input_weight', tf.cast(feed_weight, tf.float16))

# build network, on GPU by default
model = resnet.ResNet(params_model)
loss, step = model.train(feed_img, feed_gt, feed_weight)
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()

# define saver
saver_img = tf.train.Saver(var_list=get_imgnet_var())
saver_parent = tf.train.Saver()

# run the session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)
    iter_50_handle = sess.run(iter_50.string_handle())
    iter_80_handle = sess.run(iter_80.string_handle())
    iter_100_handle = sess.run(iter_100.string_handle())

    # restore all variables
    saver_img.restore(sess, params_model['restore_imgnet'])
    print('restored variables from {}'.format(params_model['restore_imgnet']))
    # saver_parent.restore(sess, params_model['restore_parent_bin'])
    # print('restored variables from {}'.format(params_model['restore_parent_bin']))
    # set deconv filters
    sess.run(set_conv_transpose_filters(tf.global_variables()))
    print('All weights initialized.')

    # each step choose a random scale/dataset for training
    print("Starting training for {0} epochs, {1} global steps.".format(epochs, num_frames*epochs))
    for ep in range(epochs):
        print("Epoch {} ...".format(ep))
        for local_step in range(steps_per_epochs):

            # randomly choose image scales from [0.5, 0.8, 1.0]
            rand_v = np.random.rand()
            train_scale = 0
            if rand_v <= 0.33:
                feed_dict_v = {handle: iter_50_handle}
            elif rand_v <= 0.66:
                feed_dict_v = {handle: iter_80_handle}
            else:
                feed_dict_v = {handle: iter_100_handle}

            # execute back_prop
            loss_, step_, sum_all_ = sess.run([loss, step, sum_all], feed_dict=feed_dict_v)

            # save summary
            if global_step % summary_write_interval == 0 and global_step !=0:
                sum_writer.add_summary(sum_all_, global_step)

            # print out loss to screen
            if global_step % print_screen_interval == 0:
                print("Global step {0} loss: {1}".format(global_step, loss_))

            # save .ckpt
            if global_step % save_ckpt_interval == 0 and global_step != 0:
                saver_parent.save(sess=sess,
                           save_path = params_model['save_path'],
                           global_step=global_step,
                           write_meta_graph=False)
                print('Saved checkpoint.')
            global_step += 1

    print("Finished training.")





