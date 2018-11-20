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
import numpy as np
sys.path.append("..")
from dataset import DAVIS_dataset
from core import resnet
from core.nn import get_imgnet_var
from core.nn import get_full_var
from core.nn import get_OF_Feat_Adam
from core.nn import get_main_Adam
from tensorflow.python import pywrap_tensorflow

def transf_slow(feat_arr, flow_arr):
    '''
    :param feat_arr: [h,w,c]
    :param flow_arr: [h,w,2]
    :return: trans_feat: [1,h,w,c]
    '''
    feat_shape = feat_arr.shape
    flow_shape = flow_arr.shape
    assert feat_shape[0] == flow_shape[0], "Feat_map/Flow shape does not match!"
    assert feat_shape[1] == flow_shape[1], "Feat_map/Flow shape does not match!"
    h = feat_shape[0]
    w = feat_shape[1]
    c = feat_shape[2]
    new_feat = np.zeros((h, w, c), np.float32)
    for idx_h in range(h):
        for idx_w in range(w):
            motion_h = int(round(flow_arr[idx_h][idx_w][1]))
            motion_w = int(round(flow_arr[idx_h][idx_w][0]))
            new_h = idx_h + motion_h
            new_w = idx_w + motion_w
            if new_h < h and new_h >= 0 and new_w < w and new_w >= 0:
                    new_feat[new_h][new_w] = feat_arr[idx_h][idx_w]

    new_feat = new_feat[np.newaxis, :]

    return new_feat

# parse argument
# conf_train_flag = int(sys.argv[1]) # 0 for main, 1 for OF/Feat_trans
# conf_epochs = int(sys.argv[2]) # 2 for main, 5 for OF/Feat_trans
# conf_lr = float(sys.argv[3]) # 1e-5 for main, 5e-5 for OF/Feat_trans
# conf_save_ckpt_interval = int(sys.argv[4]) # 1200(2ep) for main, 3000(5ep) for OF/Feat_trans
# conf_restore_ckpt = str(sys.argv[5]) # only change the suffix of saved ckpt file
# conf_l2 = float(sys.argv[6]) # 0.0005 for main, 0.0002 for OF/Feat_trans
# conf_tsboard_save = str(sys.argv[7])
conf_train_flag = 1
conf_epochs = 5
conf_lr = 5e-5
conf_save_ckpt_interval = 3000
conf_l2 = 0.0002
conf_tsboard_save = 'iter_1_OF'

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
    'batch': 1, # feed a random pair of images at a time
    'l2_weight': conf_l2,
    'init_lr': conf_lr,
    'data_format': 'NCHW', # optimal for cudnn
    'save_path': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large_Flowside/iter1_manul/att_bin.ckpt',
    'tsboard_logs': '../data/tsboard_logs/attention_bin/CNN-part-gate-img-v4_large_Flowside/'+conf_tsboard_save,
    # 'restore_0': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large_Flowside/'+conf_restore_ckpt,
    'restore_0': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large_Flowside/iter0_manul/att_bin.ckpt-1200',
    'restore_parent_bin': '../data/ckpts/xxx.ckpt'
}
# define epochs
epochs = conf_epochs
frames_per_seq = 100 # each seq is extended to 100 frames by padding previous frames inversely
steps_per_seq = 10 # because accumulate gradients 10 times before BP
num_seq = 60
steps_per_ep = num_seq * steps_per_seq
acc_count = 10 # accumulate 10 gradients
total_steps = epochs * steps_per_ep # total steps of BP, 60000
global_step = tf.Variable(0, name='global_step', trainable=False) # incremented automatically by 1 after 1 BP
save_ckpt_interval = conf_save_ckpt_interval # corresponds to 2 epoch
summary_write_interval = 20
print_screen_interval = 10

# define placeholders
feed_img = tf.placeholder(tf.float32, (params_model['batch']+1, None, None, 3)) # img(t+1), img(t)
feed_seg = tf.placeholder(tf.int32, (params_model['batch'], None, None, 1)) # img(t+1)
feed_weight = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # img/seg(t+1)
feed_att = tf.placeholder(tf.int32, (params_model['batch']+1, None, None, 1)) # img(t+1), img(t)
feed_flow = tf.placeholder(tf.float32, (params_model['batch'], None, None, 2)) # img(t)
feed_feat_flow_trans = []
for i in range(5):
    feed_feat_flow_trans.append(tf.placeholder(tf.float32, name="feat_trans_"+str(i)))
feed_mask = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # mask for trans_obj and surrounding background, smaller than att(t+1)
feed_mask_w = tf.placeholder(tf.float32, (params_model['batch'], None, None, 1)) # balance weight for trans_obj
feed_train_flag = tf.placeholder(tf.int32, ([])) # train_flag: 0,1,2,3


# display
sum_img1 = tf.summary.image('img_t1', feed_img[0:1,:,:,:])
sum_img0 = tf.summary.image('img_t0', feed_img[1:2,:,:,:])
sum_seg = tf.summary.image('seg_t1', tf.cast(feed_seg, tf.float16))
sum_weight = tf.summary.image('weight_t1', feed_weight)
sum_att1 = tf.summary.image('att_t1', tf.cast(feed_att[0:1,:,:,:], tf.float16))
sum_att0 = tf.summary.image('att_t0', tf.cast(feed_att[1:2,:,:,:], tf.float16))
sum_mask = tf.summary.image('mask', tf.cast(feed_mask, tf.float16))
sum_mask_w = tf.summary.image('mask_w', tf.cast(feed_mask_w, tf.float16))


# build network, on GPU by default
model = resnet.ResNet(params_model)
loss, bp_step, grad_acc_op, feat_trans_eval = model.train(feed_img, feed_seg, feed_weight, feed_att, feed_flow, feed_feat_flow_trans, feed_mask,
                                         feed_mask_w, global_step, acc_count, conf_train_flag)
sum_all = tf.summary.merge_all()
sum_train_flag = tf.summary.scalar('train_flag', feed_train_flag)


reader = pywrap_tensorflow.NewCheckpointReader(params_model['restore_0'])
if conf_train_flag == 0: # get OF/Feat_trans Adams
    dummy_adam = get_OF_Feat_Adam(reader)
elif conf_train_flag == 1: # get main Adams
    dummy_adam = get_main_Adam(reader)
init_op = tf.global_variables_initializer()

# define saver
saver_tmp = tf.train.Saver(get_imgnet_var())
saver_parent = tf.train.Saver(max_to_keep=20)

# # initialize adam betas because of bad historical ckpt
# with tf.variable_scope('', reuse=tf.AUTO_REUSE):
#     beta1 = tf.get_variable('beta1_power', shape=[], initializer=tf.constant_initializer(0.9))
#     beta2 = tf.get_variable('beta2_power', shape=[], initializer=tf.constant_initializer(0.999))
# init_betas = tf.initializers.variables([beta1, beta2], name='init_betas')

# run the session
with tf.Session(config=config_gpu) as sess:
    sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    img, seg, weight, att, flow, mask, mask_w = mydata.get_a_random_sample()
    feed_dict_v = {feed_img: img, feed_seg: seg, feed_weight: weight, feed_att: att,
                   feed_flow: flow, feed_mask: mask, feed_mask_w: mask_w}
    sess.run(init_op, feed_dict=feed_dict_v)

    # restore selected/all variables
    saver_tmp.restore(sess, params_model['restore_0'])
    print('restored variables from {}'.format(params_model['restore_0']))
    print('All weights initialized.')
    # sess.run(init_betas)

    print("Starting training for {0} epochs, {1} total steps.".format(epochs, total_steps))
    train_flag = conf_train_flag
    for ep in range(epochs):
        print("Epoch {} ...".format(ep))
        # train steps for each epoch
        for local_step in range(steps_per_ep):
            # accumulate gradients
            for _ in range(acc_count):
                # choose an image randomly (randomly pre-processing)
                img, seg, weight, att, flow, mask, mask_w = mydata.get_a_random_sample()
                feed_dict_v = {feed_img: img, feed_seg: seg, feed_weight: weight, feed_att: att,
                               feed_flow: flow, feed_mask: mask, feed_mask_w: mask_w}
                train_flag_dict = {feed_train_flag: train_flag}
                sum_flag_ = sess.run(sum_train_flag, feed_dict=train_flag_dict)
                sum_writer.add_summary(sum_flag_, global_step.eval())
                # forward
                if train_flag == 0:
                    run_result = sess.run([loss, sum_all] + grad_acc_op[0] + grad_acc_op[2], feed_dict=feed_dict_v)
                elif train_flag == 1:
                    feat_trans_pairs = sess.run(feat_trans_eval, feed_dict=feed_dict_v) # a list of np.arrays [HWC, HW2]
                    feat_transformed_dict = {}
                    for i in range(5):
                        ele = feat_trans_pairs[i]
                        feat_transformed_dict.update({feed_feat_flow_trans[i]: transf_slow(ele[0], ele[1])})
                    feed_dict_v.update(feat_transformed_dict)
                    run_result = sess.run([loss, sum_all] + grad_acc_op[1]+ grad_acc_op[2], feed_dict=feed_dict_v)
                elif train_flag == 2:
                    feat_trans_pairs = sess.run(feat_trans_eval, feed_dict=feed_dict_v)
                    feat_transformed_dict = {}
                    for i in range(5):
                        ele = feat_trans_pairs[i]
                        feat_transformed_dict.update({feed_feat_flow_trans[i]: transf_slow(ele[0], ele[1])})
                    feed_dict_v.update(feat_transformed_dict)
                    run_result = sess.run([loss, sum_all] + grad_acc_op[0] + grad_acc_op[1] + grad_acc_op[2],
                                          feed_dict=feed_dict_v)
                loss_ = run_result[0]
                sum_all_ = run_result[1]
            # BP, increment global_step by 1 automatically
            sess.run(bp_step)
            # save summary
            if global_step.eval() % summary_write_interval == 0 and global_step.eval() != 0:
                sum_writer.add_summary(sum_all_, global_step.eval())
            # print out loss to screen
            if global_step.eval() % print_screen_interval == 0:
                print("Global step {0} loss: {1}".format(global_step.eval(), loss_))
            # save .ckpt
            # if global_step.eval() % save_ckpt_interval == 0 and global_step.eval() != 0:
            #     saver_parent.save(sess=sess,
            #                       save_path=params_model['save_path'],
            #                       global_step=global_step,
            #                       write_meta_graph=False)
            #     print('Saved checkpoint.')
    print('Finished training.')
    saver_parent.save(sess=sess,
                      save_path=params_model['save_path'],
                      global_step=global_step,
                      write_meta_graph=False)
    print('Saved checkpoint on global_step {0}'.format(global_step.eval()))
