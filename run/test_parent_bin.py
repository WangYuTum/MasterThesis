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
from scipy.misc import imsave
from tensorflow.python import debug as tf_debug

# parse argument
arg_fine_tune = int(sys.argv[1])
arg_fine_tune_seq = int(sys.argv[2])


# set fine-tune or test
FINE_TUNE = arg_fine_tune
FINE_TUNE_seq = arg_fine_tune_seq # max 30
SUP = 0

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
    'mode': 'parent_finetune_binary',
    'seq_path': val_seq_paths[FINE_TUNE_seq],
}

# get seq data
with tf.device('/cpu:0'):
    val_data = DAVIS_dataset(params_data)
    if FINE_TUNE == 1:
        # list of numpy array: [img,gt,weight]. img with [H,W,3], gt with [H,W,1], weight with [H,W,1]
        train_gt_weight = val_data.get_one_shot_pair()
    else:
        test_frames = val_data.get_test_frames() # list: [img1, img2, ...]. img with [H, W, 3]
if FINE_TUNE == 1:
    print('Load fine-tune data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
    feed_one_shot_gt = tf.placeholder(tf.int32, [1, None, None, 1])
    feed_one_shot_weight = tf.placeholder(tf.float32, [1, None, None, 1])
else:
    print('Load test data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])

# config model params
if FINE_TUNE == 1:
    params_model = {
        'batch': 1,
        'l2_weight': 0.0002,
        'init_lr': 1e-5, # original paper: 1e-8, can be further tuned
        'data_format': 'NCHW', # optimal for cudnn
        'save_path': '../data/ckpts/fine-tune/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt',
        'tsboard_logs': '../data/tsboard_logs/fine-tune/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1],
        'restore_parent_bin': '../data/ckpts/parent-sup/parent_binary_train.ckpt-30000'
    }
    global_iters = 500 # original paper: 500
    save_ckpt_interval = 500
    summary_write_interval = 10
    print_screen_interval = 10
else:
    params_model = {
        'batch': 1,
        'data_format': 'NCHW',  # optimal for cudnn
        'restore_fine-tune_bin': '../data/ckpts/fine-tune/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt-500',
        'save_result_path': '../data/results/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]
    }

# display on tsboard only during fine-tuning
if FINE_TUNE == 1:
    sum_img = tf.summary.image('input_img', feed_img)
    sum_gt = tf.summary.image('input_gt', tf.cast(feed_one_shot_gt, tf.float16))
    sum_weight = tf.summary.image('input_weight', tf.cast(feed_one_shot_weight, tf.float16))

# build network, on GPU by default
model = resnet.ResNet(params_model)
if FINE_TUNE == 1:
    loss, step = model.train(feed_img, feed_one_shot_gt, feed_one_shot_weight, SUP, 1)
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver
    saver_fine_tune = tf.train.Saver()
else:
    prob_map, mask = model.test(feed_img) # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
    init_op = tf.global_variables_initializer()
    sum_all = tf.summary.merge_all()
    # define Saver
    saver_test = tf.train.Saver()

# run session
with tf.Session(config=config_gpu) as sess:
    # DEBUG
    # sess = tf_debug.LocalCLIDebugWrapperSession(sess)
    # sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)

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
    sess.run(set_conv_transpose_filters(tf.global_variables()))
    print('All weights initialized.')

    # starting fine-tuning/testing
    if FINE_TUNE == 1:
        print("Starting fine-tuning for {0}, {1} global steps.".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1],
                                                                       global_iters))
        feed_dict_v = {
            feed_img: train_gt_weight[0][np.newaxis,:],
            feed_one_shot_gt: train_gt_weight[1][np.newaxis,:],
            feed_one_shot_weight: train_gt_weight[2][np.newaxis,:]
        }
        for iter_step in range(global_iters):
            loss_, step_, sum_all_ = sess.run([loss, step, sum_all], feed_dict=feed_dict_v)
            if iter_step % summary_write_interval == 0:
                sum_writer.add_summary(sum_all_, iter_step)
            if iter_step % print_screen_interval == 0:
                print("Fine-tune step {0} loss: {1}".format(iter_step, loss_))
            if (iter_step+1) % save_ckpt_interval == 0 and iter_step != 0:
                saver_fine_tune.save(sess=sess,
                                     save_path=params_model['save_path'],
                                     global_step=iter_step+1,
                                     write_meta_graph=False)
                print('Saved checkpoint at iter {}'.format(iter_step+1))
        print("Finished fine-tuning.")

    else:
        print("Starting inference for {}".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
        for test_idx in range(len(test_frames)):
            print("Inference on frame {}".format(test_idx+1))
            feed_dict_v = {feed_img: test_frames[test_idx][np.newaxis,:]}
            # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
            prob_map_, mask_ = sess.run([prob_map, mask], feed_dict=feed_dict_v)
            save_name = str(test_idx+1).zfill(5) + '.png'
            save_path = params_model['save_result_path'] + '/' + save_name
            # save binary mask
            imsave(save_path, np.squeeze(mask_))
            # save prob map
            # imsave(save_path.replace('.png', '_prob.png'), np.squeeze(prob_map_))
            print("Saved result.")
        print("Finished inference.")









