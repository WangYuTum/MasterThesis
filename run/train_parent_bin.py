# - The Attention_bin branch train a model using sequences with binary object mask.
# - The model parameters (CNN part) are init from Master branch (generic objectness), other parameters
# are randomly init.
# - The model takes batch of 2 sequences for now. No batch_norm. Add batch_norm after success.
# - The shorter sequence is padded with zeros.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
from dataset import get_flip_bool
from dataset import get_scale
from dataset import standardize
from dataset import ge_att_pairs
from dataset import random_resize_flip
from dataset import pack_reshape_batch
from dataset import get_balance_weights
from core import resnet
from core.nn import set_conv_transpose_filters
from core.nn import get_imgnet_var

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
#config_gpu = tf.ConfigProto()
#config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.95

# config dataset params
params_data = {
    'mode': 'train',
    'seq_set': '/usr/stud/wangyu/DAVIS17_train_val/ImageSets/2017/train.txt',
}
with tf.device('/cpu:0'):
    mydata = DAVIS_dataset(params_data)

# config train params
params_model = {
    'batch': 4, # feed consecutive images at once
    'l2_weight': 0.0002,
    'init_lr': 1e-5, # original paper: 1e-8,
    'l1_att': 1e-4, # l1 sparsity on attention map
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/attention_bin/att_bin.ckpt',
    'tsboard_logs': '../data/tsboard_logs/attention_bin/',
    'restore_imgnet': '../data/ckpts/imgnet.ckpt', # restore model from where
    'restore_parent_bin': '../data/ckpts/attention_bin/att_bin.ckpt-xxx'
}
# define epochs
epochs = 10
frames_per_seq = 100
steps_per_seq = 33
num_seq = 60
total_steps = epochs * num_seq * steps_per_seq
global_step = tf.Variable(0, name='global_step', trainable=False) # incremented automatically by 1 after each apply_gradients
acc_count = 1
save_ckpt_interval = 10000
summary_write_interval = 2 # 50
print_screen_interval = 1 # 20

# define placeholders
feed_img = tf.placeholder(tf.float32, (params_model['batch'], None, None, 3)) # f0, f1, f2, f3
feed_seg = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1)) # s0, s1, s2, s3
feed_weight = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # w_s0, w_att1, w_s2, w_att3
feed_att = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1)) # a0, a1, a2, a3
feed_att_oracle = tf.placeholder(tf.int32, (2, None, None, 1)) # a01, a23

# display
sum_img0 = tf.summary.image('img0', feed_img[0:1,:,:,:])
sum_img1 = tf.summary.image('img1', feed_img[1:2,:,:,:])
sum_img2 = tf.summary.image('img2', feed_img[2:3,:,:,:])
sum_img3 = tf.summary.image('img3', feed_img[3:4,:,:,:])
sum_att0 = tf.summary.image('f0_att', tf.cast(feed_att[0:1,:,:,:], tf.float16)) # to gate f0
sum_att01 = tf.summary.image('f01_att', tf.cast(feed_att_oracle[0:1,:,:,:], tf.float16)) # to gate f1
sum_att1 = tf.summary.image('f1_att', tf.cast(feed_att[1:2,:,:,:], tf.float16)) # to prediction a1
sum_att2 = tf.summary.image('f2_att', tf.cast(feed_att[2:3,:,:,:], tf.float16)) # to gate f2
sum_att23 = tf.summary.image('f23_att', tf.cast(feed_att_oracle[1:2,:,:,:], tf.float16)) # to gate f3
sum_att3 = tf.summary.image('f3_att', tf.cast(feed_att[3:4,:,:,:], tf.float16)) # to prediction a3
sum_seg0 = tf.summary.image('f0_seg', tf.cast(feed_seg[0:1,:,:,:], tf.float16)) # to prediction s0
sum_seg1 = tf.summary.image('f1_seg', tf.cast(feed_seg[1:2,:,:,:], tf.float16))
sum_seg2 = tf.summary.image('f2_seg', tf.cast(feed_seg[2:3,:,:,:], tf.float16)) # to prediction s2
sum_seg3 = tf.summary.image('f3_seg', tf.cast(feed_seg[3:4,:,:,:], tf.float16))
sum_w_s0 = tf.summary.image('weight_s0', tf.cast(feed_weight[0:1,:,:,:], tf.float16))
sum_w_att1 = tf.summary.image('weight_att1', tf.cast(feed_weight[1:2,:,:,:], tf.float16))
sum_w_s2 = tf.summary.image('weight_s2', tf.cast(feed_weight[2:3,:,:,:], tf.float16))
sum_w_att3 = tf.summary.image('weight_att3', tf.cast(feed_weight[3:4,:,:,:], tf.float16))

# build network, on GPU by default
model = resnet.ResNet(params_model)
loss, bp_step = model.train(feed_img, feed_seg, feed_weight, feed_att, feed_att_oracle, acc_count, global_step)
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()

# define saver
saver_img = tf.train.Saver(var_list=get_imgnet_var())
saver_parent = tf.train.Saver()

# run the session
with tf.Session() as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)

    # restore all variables
    saver_img.restore(sess, params_model['restore_imgnet'])
    print('restored variables from {}'.format(params_model['restore_imgnet']))
    # set deconv filters
    sess.run(set_conv_transpose_filters(tf.global_variables()))
    print('All weights initialized.')

    print("Starting training for {0} epochs, {1} total steps.".format(epochs, total_steps))
    for ep in range(epochs):
        print("Epoch {} ...".format(ep))
        # train num_seq for each epoch
        for rand_seq in range(num_seq):
            # get a random seq, rescale/flip flag
            seq_imgs, seq_segs = mydata.get_random_seq() # imgs: [img0, img1, ...] segs: [seg0, seg1, ...]
            flip_bool = get_flip_bool()
            scale_f = get_scale()
            for local_step in range(steps_per_seq):
                # prepare data for a local step
                f0 = seq_imgs[local_step]
                f1 = seq_imgs[local_step+1]
                f2 = seq_imgs[local_step+2]
                f3 = seq_imgs[local_step+3]
                s0 = seq_segs[local_step]
                s1 = seq_segs[local_step+1]
                s2 = seq_imgs[local_step+2]
                s3 = seq_imgs[local_step+3]
                f0, f1, f2, f3, s0, s1, s2, s3 = standardize(f0, f1, f2, f3, s0, s1, s2, s3) # dtype converted
                a0, a1, a2, a3, a01, a23 = ge_att_pairs(s0, s1, s2, s3)
                # resize/flip same for all frames to the current seq, returned all data has shape [H,W,C]
                f0, f1, f2, f3, s0, s1, s2, s3, a0, a1, a2, a3, a01, a23 = random_resize_flip(f0, f1, f2, f3,
                                                                                              s0, s1, s2, s3,
                                                                                              a0, a1, a2, a3,
                                                                                              a01, a23,
                                                                                              flip_bool, scale_f)
                w_s0, w_att1, w_s2, w_att3 = get_balance_weights(s0, a0, a1, s2, a2, a3)
                feed_dict_v = {feed_img: pack_reshape_batch(f0, f1, f2, f3),
                               feed_seg: pack_reshape_batch(s0, s1, s2, s3),
                               feed_weight: pack_reshape_batch(w_s0, w_att1, w_s2, w_att3),
                               feed_att: pack_reshape_batch(a0, a1, a2, a3),
                               feed_att_oracle: np.stack((a01, a23), axis=0)}
                # compute loss and gradients
                run_result = sess.run([loss, sum_all], feed_dict=feed_dict_v)
                loss_ = run_result[0]
                sum_all_ = run_result[1]
                # execute BP, increment global_step by 1 automatically
                sess.run(bp_step)

                # save summary
                if global_step.eval() % summary_write_interval == 0 and global_step.eval() != 0:
                    sum_writer.add_summary(sum_all_, global_step.eval())
                # print out loss to screen
                if global_step.eval() % print_screen_interval == 0:
                    print("Global step {0} loss: {1}".format(global_step.eval(), loss_))
                # save .ckpt
                if global_step.eval() % save_ckpt_interval == 0 and global_step.eval() != 0:
                    saver_parent.save(sess=sess,
                                      save_path=params_model['save_path'],
                                      global_step=global_step,
                                      write_meta_graph=False)
                    print('Saved checkpoint.')

    print('Finished training.')









