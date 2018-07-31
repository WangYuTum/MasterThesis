# - The Attention_bin branch train a model using sequences with binary object mask.
# - The model parameters (CNN part) are init from Master branch (generic objectness), other parameters
# are randomly init.
# - The model takes batch of 2 sequences for now. No batch_norm. Add batch_norm after success.
# - The shorter sequence is padded with zeros.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
from core import resnet
from core.nn import get_imgnet_var
import numpy as np

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# config dataset params
params_data = {
    'mode': 'train',
    'seq_set': '/usr/stud/wangyu/DAVIS17_train_val/ImageSets/2017/train.txt',
}
with tf.device('/cpu:0'):
    mydata = DAVIS_dataset(params_data)

# config train params
params_model = {
    'batch': 1, # feed a random image at a time
    'l2_weight': 0.0002,
    'init_lr': 1e-5, # original paper: 1e-8,
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_small/att_bin.ckpt',
    'tsboard_logs': '../data/tsboard_logs/attention_bin/CNN-part-gate-img-v4_small',
    'restore_imgnet': '../data/ckpts/imgnet.ckpt', # restore model from where
    'restore_parent_bin': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_small/att_bin.ckpt-xxx'
}
# define epochs
epochs = 100
frames_per_seq = 100 # each seq is extended to 100 frames by padding previous frames inversely
steps_per_seq = 10 # because accumulate gradients 10 times before BP
num_seq = 60
steps_per_ep = num_seq * steps_per_seq
acc_count = 10 # accumulate 10 gradients
total_steps = epochs * steps_per_ep # total steps of BP, 60000
global_step = tf.Variable(0, name='global_step', trainable=False) # incremented automatically by 1 after 1 BP
save_ckpt_interval = 12000 # corresponds to 20 epoch
summary_write_interval = 50
print_screen_interval = 20

# define placeholders
feed_img = tf.placeholder(tf.float32, (params_model['batch'], None, None, 3))
feed_seg = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1))
feed_weight = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1))
feed_att = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1))

# display
sum_img = tf.summary.image('img', feed_img)
sum_seg = tf.summary.image('seg', tf.cast(feed_seg, tf.float16))
sum_w = tf.summary.image('weight', feed_weight)
sum_att = tf.summary.image('att', tf.cast(feed_att, tf.float16))


# build network, on GPU by default
model = resnet.ResNet(params_model)
loss, bp_step, grad_acc_op = model.train(feed_img, feed_seg, feed_weight, feed_att, global_step, acc_count)
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()

# define saver
saver_img = tf.train.Saver(var_list=get_imgnet_var())
saver_parent = tf.train.Saver(max_to_keep=10)

# run the session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)

    # restore all variables
    saver_img.restore(sess, params_model['restore_imgnet'])
    print('restored variables from {}'.format(params_model['restore_imgnet']))
    print('All weights initialized.')

    print("Starting training for {0} epochs, {1} total steps.".format(epochs, total_steps))
    for ep in range(epochs):
        print("Epoch {} ...".format(ep))
        # train steps for each epoch
        for local_step in range(steps_per_ep):
            # accumulate gradients
            acc_i = 0
            while acc_i < acc_count:
                # choose an image randomly (randomly pre-processing)
                img, seg, weight, att = mydata.get_a_random_sample() # [1,h,w,3] float32, [1,h,w,1] int32, [1,h,w,1] float32
                if np.count_nonzero(att) < 10:
                    continue
                feed_dict_v = {feed_img: img, feed_seg: seg, feed_weight: weight, feed_att: att}
                # forward
                run_result = sess.run([loss, sum_all]+grad_acc_op, feed_dict=feed_dict_v)
                loss_ = run_result[0]
                sum_all_ = run_result[1]
                acc_i = acc_i + 1
            # BP, increment global_step by 1 automatically
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

