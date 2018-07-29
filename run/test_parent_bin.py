from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
sys.path.append("..")
from dataset import DAVIS_dataset
from core import resnet
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure

# parse argument
arg_fine_tune = int(sys.argv[1])
arg_fine_tune_seq = int(sys.argv[2])


# set fine-tune or test
FINE_TUNE = arg_fine_tune
FINE_TUNE_seq = arg_fine_tune_seq # max 30

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True
#config_gpu.gpu_options.per_process_gpu_memory_fraction = 0.6

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
        test_frames = val_data.get_test_frames() # list: [[img1, att1], [img2, att2], ...], img with [H,W,3], att [H,W,1]
if FINE_TUNE == 1:
    print('Load fine-tune data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
    feed_one_shot_gt = tf.placeholder(tf.int32, [1, None, None, 1])
    feed_one_shot_weight = tf.placeholder(tf.float32, [1, None, None, 1])
    feed_one_shot_att = tf.placeholder(tf.int32, [1, None, None, 1])
else:
    print('Load test data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
    feed_att = tf.placeholder(tf.int32, [1, None, None, 1])

# config model params
if FINE_TUNE == 1:
    params_model = {
        'batch': 1,
        'l2_weight': 0.0002,
        'init_lr': 1e-6, # original paper: 1e-8, can be further tuned
        'data_format': 'NCHW', # optimal for cudnn
        'save_path': '../data/ckpts/fine-tune/attention_bin/CNN-part-gate-img-v4_large/80ep/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt',
        'tsboard_logs': '../data/tsboard_logs/fine-tune/attention_bin/CNN-part-gate-img-v4_large/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1],
        'restore_parent_bin': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large/att_bin.ckpt-48000'
    }
    global_iters = 1000 # original paper: 500
    save_ckpt_interval = 100
    summary_write_interval = 10
    print_screen_interval = 10
    acc_count = 1
    global_step = tf.Variable(0, name='global_step',
                              trainable=False)  # incremented automatically by 1 after each apply_gradients
else:
    params_model = {
        'batch': 1,
        'data_format': 'NCHW',  # optimal for cudnn
        'restore_fine-tune_bin': '../data/ckpts/fine-tune/attention_bin/CNN-part-gate-img-v4_large/80ep/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt-48700',
        'save_result_path': '../data/results/iter1000/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]
    }

# display on tsboard only during fine-tuning
if FINE_TUNE == 1:
    sum_img = tf.summary.image('input_img', feed_img)
    sum_gt = tf.summary.image('input_gt', tf.cast(feed_one_shot_gt, tf.float16))
    sum_weight = tf.summary.image('input_weight', tf.cast(feed_one_shot_weight, tf.float16))
    sum_att = tf.summary.image('input_att', tf.cast(feed_one_shot_att, tf.float16))

# build network, on GPU by default
model = resnet.ResNet(params_model)
if FINE_TUNE == 1:
    loss, bp_step, grad_acc_op = model.train(feed_img, feed_one_shot_gt,
                                             feed_one_shot_weight, feed_one_shot_att,
                                             global_step, acc_count)
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver
    saver_fine_tune = tf.train.Saver(max_to_keep=11)
else:
    prob_map, mask = model.test(feed_img, feed_att) # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver
    saver_test = tf.train.Saver()

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
    # set deconv filters
    print('All weights initialized.')

    # starting fine-tuning/testing
    if FINE_TUNE == 1:
        print("Starting fine-tuning for {0}, {1} global steps.".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1],
                                                                       global_iters))
        for iter_step in range(global_iters):
            feed_dict_v = {
                feed_img: train_gt_weight[0][np.newaxis, :],
                feed_one_shot_gt: train_gt_weight[1][np.newaxis, :],
                feed_one_shot_weight: train_gt_weight[2][np.newaxis, :],
                feed_one_shot_att: train_gt_weight[3][np.newaxis, :]
            }
            # compute loss and acc grad
            run_result = sess.run([loss, sum_all] + grad_acc_op, feed_dict=feed_dict_v)
            loss_ = run_result[0]
            sum_all_ = run_result[1]
            # execute BP
            _ = sess.run(bp_step)
            # Get the train_gt_weight for the 0-th frame, but with randomized size/shift
            train_gt_weight = val_data.get_one_shot_pair()

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
        for test_idx in range(len(test_frames)):
            print("Inference on frame {}".format(test_idx+1))
            feed_dict_v = {feed_img: test_frames[test_idx][0][np.newaxis,:],
                           feed_att: test_frames[test_idx][1][np.newaxis,:]}
            # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
            prob_map_, mask_ = sess.run([prob_map, mask], feed_dict=feed_dict_v)
            save_name = str(test_idx+1).zfill(5) + '.png'
            save_path = params_model['save_result_path'] + '/' + save_name
            # save binary mask
            final_seg = np.multiply(np.squeeze(test_frames[test_idx][1]), np.squeeze(mask_))
            imsave(save_path, final_seg)
            # save prob map
            # imsave(save_path.replace('.png', '_prob.png'), np.squeeze(prob_map_))
            print("Saved result.")
        print("Finished inference.")









