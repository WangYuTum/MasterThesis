from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
from core import resnet
from core.nn import get_two_stream_var
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure


# parse argument
arg_fine_tune = int(sys.argv[1])
arg_fine_tune_seq = int(sys.argv[2])

# set fine-tune or test
FINE_TUNE = 0
FINE_TUNE_seq = arg_fine_tune_seq # max 30

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.6

# get all val seq paths
val_seq_txt = '../../../DAVIS17_train_val/ImageSets/2017/val.txt'
val_seq_paths = []
with open(val_seq_txt) as t:
    val_seq_names = t.read().splitlines() # e.g. [bike-packing, blackswan, ...]
for i in range(len(val_seq_names)):
    val_seq_path = os.path.join('../../../DAVIS17_train_val/JPEGImages/480p', val_seq_names[i])
    val_seq_paths.append(val_seq_path)
print('Got {} val seqs in total.'.format(len(val_seq_paths)))

# config dataset params (480p images)
params_data = {
    'mode': 'val',
    'seq_path': val_seq_paths[FINE_TUNE_seq],
}

# get seq data
with tf.device('/cpu:0'):
    val_data = DAVIS_dataset(params_data)
    if FINE_TUNE == 1:
        # list of numpy array: [img,gt,weight]. img with [H,W,3], gt with [H,W,1], weight with [H,W,1]
        train_gt_weight = val_data.get_one_shot_pair()
    else:
        first_pair = val_data.get_one_shot_pair()
        test_frames = val_data.get_test_frames() # list: [img1, img2, ...]. img with [H, W, 3]
        num_test_frames = len(test_frames)

# define feed data to tf.Session
if FINE_TUNE == 1:
    print('Load fine-tune data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
    feed_one_shot_gt = tf.placeholder(tf.int32, [1, None, None, 1])
    feed_one_shot_weight = tf.placeholder(tf.float32, [1, None, None, 1])
else:
    print('Load test data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [2, None, None, 3]) # I_t, I_t+1
    feed_att = tf.placeholder(tf.int32, [1, None, None, 1]) # attention on I_t
    feed_att_oracle = tf.placeholder(tf.int32, [1, None, None, 1]) # attention oracle on I_t+1
    feed_state = tf.placeholder(tf.float32, [2, 30, 56, None])  # Cell [1,30,56,128], Hidden [1,30,56,128]

# config model params
# TODO
if FINE_TUNE == 1:
    params_model = {
        'batch': 2,
        'l2_weight': 0.0002,
        'init_lr': 1e-5, # original paper: 1e-8, can be further tuned
        'data_format': 'NHWC', # optimal for cudnn
        'save_path': '../data/ckpts/fine-tune/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt',
        'tsboard_logs': '../data/tsboard_logs/fine-tune/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1],
        'restore_parent_bin': '../data/ckpts/parent-sup-no-bn/parent_binary_train.ckpt-90000'
    }
    global_iters = 500 # original paper: 500
    save_ckpt_interval = 500
    summary_write_interval = 10
    print_screen_interval = 10
    global_step = tf.Variable(0, name='global_step',
                              trainable=False)  # incremented automatically by 1 after each apply_gradients
else:
    params_model = {
        'batch': 2,
        'data_format': 'NHWC',  # optimal for cudnn
        'restore_fine-tune_bin': '../data/ckpts/attention_bin/two-stream-complete/att_bin.ckpt-45000',
        # 'restore_fine-tune_bin': '../data/ckpts/fine-tune/attention_bin/two-stream-complete/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt-90500',
        'save_result_path': '../data/results/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]
    }

# display on tsboard only during fine-tuning
# TODO
if FINE_TUNE == 1:
    sum_img = tf.summary.image('input_img', feed_img)
    sum_gt = tf.summary.image('input_gt', tf.cast(feed_one_shot_gt, tf.float16))
    sum_weight = tf.summary.image('input_weight', tf.cast(feed_one_shot_weight, tf.float16))

# build network, on GPU by default
model = resnet.ResNet(params_model)
# TODO
if FINE_TUNE == 1:
    loss, step, grad_acc_op = model.train(feed_img, feed_one_shot_gt, feed_one_shot_weight, global_step)
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver
    saver_fine_tune = tf.train.Saver()
else:
    pred_mask_att, pred_mask_seg, current_state, assign_state_ops = model.test(feed_img,
                                                             feed_att,
                                                             feed_att_oracle,
                                                             feed_state) # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver, restore all variables except c_var, h_var
    saver_test = tf.train.Saver(get_two_stream_var(params_model['restore_fine-tune_bin']))

# run session
with tf.Session(config=config_gpu) as sess:

    if FINE_TUNE == 1:
        sum_writer = tf.summary.FileWriter(params_model['tsboard_logs'], sess.graph)
    sess.run(init_op)
    # restore all variables
    if FINE_TUNE == 1:
        saver_fine_tune.restore(sess, params_model['restore_parent_bin'])
        print('restored variables from {}'.format(params_model['restore_parent_bin']))
    else:
        saver_test.restore(sess, params_model['restore_fine-tune_bin'])
        print('restored variables from {}'.format(params_model['restore_fine-tune_bin']))
    print('All weights initialized.')

    # starting fine-tuning/testing
    # TODO
    if FINE_TUNE == 1:
        print("Starting fine-tuning for {0}, {1} global steps.".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1],
                                                                       global_iters))
        feed_dict_v = {
            feed_img: train_gt_weight[0][np.newaxis,:],
            feed_one_shot_gt: train_gt_weight[1][np.newaxis,:],
            feed_one_shot_weight: train_gt_weight[2][np.newaxis,:]
        }
        for iter_step in range(global_iters):
            # acc gradient
            run_result = sess.run([loss, sum_all] + grad_acc_op, feed_dict=feed_dict_v)
            loss_ = run_result[0]
            sum_all_ = run_result[1]
            # execute BP
            sess.run(step)

            if global_step.eval() % summary_write_interval == 0:
                sum_writer.add_summary(sum_all_, global_step.eval())
            if global_step.eval() % print_screen_interval == 0:
                print("Fine-tune step {0} loss: {1}".format(global_step.eval(), loss_))
            if global_step.eval() % save_ckpt_interval == 0 and global_step.eval() != 0:
                saver_fine_tune.save(sess=sess,
                                     save_path=params_model['save_path'],
                                     global_step=global_step,
                                     write_meta_graph=False)
                print('Saved checkpoint at iter {}'.format(global_step.eval()))
        print("Finished fine-tuning.")
    else:
        print("Starting inference for {}".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
        num_frames = len(test_frames)
        struct1 = generate_binary_structure(2, 2)
        current_state_val = None
        current_att_val = None
        next_att_val = None
        current_att_oracle_val = None
        for test_idx in range(num_frames):
            if test_idx == 0:
                # 1st time feed state all zeros, gt attention on I_0, lstm state init to 0s automatically
                zero_state = np.zeros((2,30,56,128), dtype=np.float32) # shape depends on lstm definition in resnet.py
                f0 = first_pair[0][np.newaxis,:]
                f1 = test_frames[0][np.newaxis,:]
                img_pair = np.concatenate((f0, f1), axis=0)
                print('feed img shape: {}'.format(img_pair.shape))
                s0 = np.squeeze(first_pair[1]) # np.uint8, [H,W]
                a0 = binary_dilation(s0, structure=struct1, iterations=30).astype(s0.dtype)
                a01 = binary_dilation(s0, structure=struct1, iterations=40).astype(s0.dtype)
                a0 = a0[np.newaxis, ..., np.newaxis].astype(np.int32)
                a01 = a01[np.newaxis, ..., np.newaxis].astype(np.int32)
                feed_dict_v = {feed_img: img_pair,
                               feed_att: a0,
                               feed_att_oracle: a01}
                att_pred_, seg_pred_, final_state_ = sess.run([pred_mask_att, pred_mask_seg, current_state],
                                                              feed_dict=feed_dict_v)
                # att_pred_, seg_pred_: [1,H,W], np.int32
                # final_state_: a tuple each with shape [1,30,56,128]
                current_state_val = np.concatenate((final_state_[0], final_state_[1]), axis=0) # [2,30,56,128]
                next_att_val = att_pred_[..., np.newaxis] # [1,H,W,1]
                att_pred_ = np.squeeze(att_pred_) # [H,W]
                current_att_oracle_val = binary_dilation(att_pred_, structure=struct1, iterations=10).astype(att_pred_.dtype)
                current_att_oracle_val = current_att_oracle_val[np.newaxis, ..., np.newaxis].astype(np.int32) # [1,H,W,1]
                # Save prediction
                save_seg_name = str(test_idx).zfill(5) + '.png'
                save_seg_path = params_model['save_result_path'] + '/' + save_seg_name
                seg_pred_gated = np.multiply(att_pred_, np.squeeze(a0))
                imsave(save_seg_path, np.squeeze(seg_pred_gated))
                save_att_name = str(test_idx+1).zfill(5) + '.png'
                save_att_path = params_model['save_result_path'] + '/att/' + save_att_name
                imsave(save_att_path, np.squeeze(att_pred_))
                print('Saved result.')
                current_att_val = next_att_val
            else:
                # feed previous state, previous generated attention
                if test_idx != num_frames-1:
                    # restore lstm state values
                    _ = sess.run(assign_state_ops, feed_dict={feed_state: current_state_val})
                    f0 = test_frames[test_idx][np.newaxis, :]
                    f1 = test_frames[test_idx+1][np.newaxis, :]
                    img_pair = np.concatenate((f0, f1), axis=0)
                    feed_dict_v = {feed_img: img_pair,
                                   feed_att: current_att_val,
                                   feed_att_oracle: current_att_oracle_val}
                    att_pred_, seg_pred_, final_state_ = sess.run([pred_mask_att, pred_mask_seg, current_state],
                                                                  feed_dict=feed_dict_v)
                    current_state_val = np.concatenate((final_state_[0], final_state_[1]), axis=0) # [2,30,56,128]
                    next_att_val = att_pred_[..., np.newaxis]  # [1,H,W,1]
                    att_pred_ = np.squeeze(att_pred_)  # [H,W]
                    current_att_oracle_val = binary_dilation(att_pred_, structure=struct1, iterations=10).astype(att_pred_.dtype)
                    current_att_oracle_val = current_att_oracle_val[np.newaxis, ..., np.newaxis].astype(np.int32)  # [1,H,W,1]
                    # Save prediction
                    save_seg_name = str(test_idx).zfill(5) + '.png'
                    save_seg_path = params_model['save_result_path'] + '/' + save_seg_name
                    seg_pred_gated = np.multiply(att_pred_, np.squeeze(current_att_val))
                    imsave(save_seg_path, np.squeeze(seg_pred_gated))
                    save_att_name = str(test_idx+1).zfill(5) + '.png'
                    save_att_path = params_model['save_result_path'] + '/att/' + save_att_name
                    imsave(save_att_path, np.squeeze(att_pred_))
                    print('Saved result.')
                    current_att_val = next_att_val
                else: # the last frame, repeat the last frame
                    # restore lstm state values
                    _ = sess.run(assign_state_ops, feed_dict={feed_state: current_state_val})
                    f0 = test_frames[test_idx][np.newaxis, :]
                    img_pair = np.concatenate((f0, f0), axis=0)
                    feed_dict_v = {feed_img: img_pair,
                                   feed_att: current_att_val,
                                   feed_att_oracle: current_att_oracle_val}
                    att_pred_, seg_pred_, final_state_ = sess.run([pred_mask_att, pred_mask_seg, current_state],
                                                                  feed_dict=feed_dict_v)
                    # Save prediction
                    save_seg_name = str(test_idx).zfill(5) + '.png'
                    save_seg_path = params_model['save_result_path'] + '/' + save_seg_name
                    seg_pred_gated = np.multiply(att_pred_, np.squeeze(current_att_val))
                    imsave(save_seg_path, np.squeeze(seg_pred_gated))
                    print('Saved result.')
        print("Finished inference.")









