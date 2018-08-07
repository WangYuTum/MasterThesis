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
from core.nn import get_parent_var

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# config dataset params
params_data = {
    'mode': 'train',
    'seq_set': '/usr/stud/wangyu/DAVIS17_train_val/ImageSets/2017/train.txt',
}
with tf.device('/cpu:0'):
    mydata = DAVIS_dataset(params_data) ### TODO: make sure each batch has the same image size

# config train params
params_model = {
    'batch': 8, # feed 2 seqs, 4 frames for each seqs
    'l2_weight': 0.0002,
    'init_lr': 1e-5, # original paper: 1e-8,
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/attention_bin/CNN-part-gate-img-v6_lstm/att_bin.ckpt',
    'tsboard_logs': '../data/tsboard_logs/attention_bin/CNN-part-gate-img-v6_lstm',
    'restore_parent': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large/att_bin.ckpt-24000', # restore model from where
    'restore_parent_bin': '../data/ckpts/attention_bin/CNN-part-gate-img-v6_lstm/att_bin.ckpt-xxx'
}
# define epochs
epochs = 100
frames_per_seq = 100 # each seq is extended to 100 frames by padding previous frames inversely
num_seq = 60
steps_per_ep = num_seq * frames_per_seq / params_model['batch'] # 750 if batch=4*2
total_steps = epochs * steps_per_ep # total steps of BP, 60000
global_step = tf.Variable(0, name='global_step', trainable=False) # incremented automatically by 1 after 1 BP
save_ckpt_interval = 7500 # corresponds to 10 epoch
summary_write_interval = 10
print_screen_interval = 5

# define placeholders
feed_img = tf.placeholder(tf.float32, (params_model['batch'], None, None, 3)) # [batch, H, W, 3]
feed_seg = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1)) # [batch, H, W, 1]
feed_weight = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # [batch, H, W, 1]
feed_att = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1)) # [batch, H, W, 1]
feed_prob = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # [batch, H, W, 1], prob of being positive features
feed_bb = tf.placeholder(tf.int32, (params_model['batch'],4)) # each image has 4 params

# display
sum_img = tf.summary.image('img', feed_img)
sum_seg = tf.summary.image('seg', tf.cast(feed_seg, tf.float16))
sum_w = tf.summary.image('weight', feed_weight)
sum_att = tf.summary.image('att', tf.cast(feed_att, tf.float16))
sum_prob = tf.summary.image('prob', tf.cast(feed_prob, tf.float16))


# build network, on GPU by default
model = resnet.ResNet(params_model)
loss, bp_step = model.train(feed_img, feed_seg, feed_weight, feed_att, feed_prob, feed_bb, global_step)
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()

# define saver
saver_parent = tf.train.Saver(var_list=get_parent_var()) # parent cnn not need to be trained, hence not restore Adam state
saver_lstm = tf.train.Saver(max_to_keep=11) # save weight every 10 ep, up to 100 ep

# run the session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)

    # restore all variables
    saver_parent.restore(sess, params_model['restore_parent'])
    print('restored variables from {}'.format(params_model['restore_parent']))
    print('All weights initialized.')

    print("Starting training for {0} epochs, {1} total steps.".format(epochs, total_steps))
    for ep in range(epochs):
        print("Epoch {} ...".format(ep))
        # train steps for each epoch
        permute_list = mydata.permute_seq_order()
        for local_step in range(steps_per_ep):
            # choose two seqs in the shuffle list, 4 frames per seq
            # TODO: get seq frames
            imgs, segs, weights, atts, prob_maps, bbs = mydata.get_batch_sample(permute_list, local_step)
            #img, seg, weight, att = mydata.get_a_random_sample() # [1,h,w,3] float32, [1,h,w,1] int32, [1,h,w,1] float32
            feed_dict_v = {feed_img: imgs, feed_seg: segs, feed_weight: weights,
                           feed_att: atts, feed_prob: prob_maps, feed_bb: bbs}
            # forward and backward
            run_result = sess.run([loss, sum_all, bp_step], feed_dict=feed_dict_v)
            loss_ = run_result[0]
            sum_all_ = run_result[1]
            # save summary
            if global_step.eval() % summary_write_interval == 0 and global_step.eval() != 0:
                sum_writer.add_summary(sum_all_, global_step.eval())
            # print out loss to screen
            if global_step.eval() % print_screen_interval == 0:
                print("Global step {0} loss: {1}".format(global_step.eval(), loss_))
            # save .ckpt
            if global_step.eval() % save_ckpt_interval == 0 and global_step.eval() != 0:
                saver_lstm.save(sess=sess,
                                  save_path=params_model['save_path'],
                                  global_step=global_step,
                                  write_meta_graph=False)
                print('Saved checkpoint.')
    print('Finished training.')

