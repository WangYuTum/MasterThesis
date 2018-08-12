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
import glob
from PIL import Image
from core.nn import get_main_var
from core.nn import get_lstm_var
TAG_FLOAT = 202021.25

def read_flow(file):

    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = np.fromfile(f, np.int32, count=1)
    h = np.fromfile(f, np.int32, count=1)
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))
    f.close()

    return flow

def transf_slow(seg_arr, flow_arr):
    seg_shape = seg_arr.shape
    flow_shape = flow_arr.shape
    assert seg_shape[0] == flow_shape[0], "Seg/Flow shape does not match!"
    assert seg_shape[1] == flow_shape[1], "Seg/Flow shape does not match!"
    h = seg_shape[0]
    w = seg_shape[1]
    new_seg = np.zeros((h, w), np.uint8)
    for idx_h in range(h):
        for idx_w in range(w):
            if seg_arr[idx_h][idx_w] == 1:
                motion_h = int(round(flow_arr[idx_h][idx_w][1]))
                motion_w = int(round(flow_arr[idx_h][idx_w][0]))
                new_h = idx_h + motion_h
                new_w = idx_w + motion_w
                if new_h < h and new_h >= 0 and new_w < w and new_w >= 0:
                    new_seg[new_h][new_w] = 1

    return new_seg

def overlay_seg_rgb(seg_arr, rgb_obj):

    seg_arr = seg_arr.astype(np.uint8)
    img_palette = np.array([0, 0, 0, 150, 0, 0])
    seg_color = Image.fromarray(seg_arr, mode='P')
    seg_color.putpalette(img_palette)
    overlay_mask = Image.fromarray(seg_arr * 150, mode='L')

    com_img = Image.composite(seg_color, rgb_obj, overlay_mask)

    return com_img

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
    test_frames = val_data.get_test_frames() # list: [[img1, att1], [img2, att2], ...], img with [H,W,3], att [H,W,1]
    print('Load test data for seq {} done.'.format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))
    feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
    feed_att = tf.placeholder(tf.int32, [1, None, None, 1])
    feed_prob = tf.placeholder(tf.float32, [1, None, None, 1])
    feed_bb = tf.placeholder(tf.int32, [4])
    feed_state = tf.placeholder(tf.float32, [2, 256, 512, 64])

# config model params
params_model = {
    'batch': 1,
    'data_format': 'NCHW',  # optimal for cudnn
    'restore_fine-tune_main': '../data/ckpts/fine-tune/attention_bin/CNN-part-gate-img-v4_large/40ep/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]+'/fine-tune.ckpt-24300',
    #'restore_fine-tune_main': '../data/ckpts/attention_bin/CNN-part-gate-img-v4_large/att_bin.ckpt-24000',
    'restore_lstm': '../data/ckpts/attention_bin/CNN-part-gate-img-v6_lstm/att_bin.ckpt-48000',
    'save_result_path': '../data/results/flow_att_seg/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1],
    'save_prob_path': '../data/results/prob_map/'+val_seq_paths[FINE_TUNE_seq].split('/')[-1]
}

# build network, on GPU by default
model = resnet.ResNet(params_model)
init_prob_map, init_seg_mask, final_seg_mask, assign_state_ops, current_state, init_lstm_inf = model.test(feed_img, feed_att, feed_prob, feed_bb, feed_state)
init_op = tf.global_variables_initializer()
# define Saver
saver_main = tf.train.Saver(var_list=get_main_var())
saver_lstm = tf.train.Saver(var_list=get_lstm_var())

# run session
with tf.Session(config=config_gpu) as sess:

    sess.run(init_op)
    # restore all variables
    saver_main.restore(sess, params_model['restore_fine-tune_main'])
    print('restored main_cnn variables from {}'.format(params_model['restore_fine-tune_main']))
    saver_lstm.restore(sess, params_model['restore_lstm'])
    print('restored lstm variables from {}'.format(params_model['restore_lstm']))
    print('All weights initialized.')

    # start test
    print("Starting inference for {}".format(val_seq_paths[FINE_TUNE_seq].split('/')[-1]))

    ### Get optical flow for the sequence
    flow_search = os.path.join(val_seq_paths[FINE_TUNE_seq].replace('JPEGImages', 'Flow'), '*.flo')
    flow_files = glob.glob(flow_search)
    flow_files.sort()
    ### Get image lists of the current sequence
    img_search = os.path.join(val_seq_paths[FINE_TUNE_seq], '*.jpg')
    img_files = glob.glob(img_search)
    img_files.sort()
    seg0_file = os.path.join(val_seq_paths[FINE_TUNE_seq].replace('JPEGImages', 'Annotations'), '00000.png')
    seg0_arr = np.greater(np.array(Image.open(seg0_file)), 0).astype(np.uint8)

    ### for the 0th frame, use it to initialize lstm state
    struct1 = generate_binary_structure(2, 2)
    att_flow = binary_dilation(seg0_arr, structure=struct1, iterations=30).astype(seg0_arr.dtype)
    att_obj = Image.fromarray(np.squeeze(att_flow.astype(np.uint8)))
    bb_params = att_obj.getbbox()  # [w1,h1,w2,h2]
    bb = [0] * 4
    bb[0] = bb_params[1]  # offset_h
    bb[1] = bb_params[0]  # offset_w
    bb[2] = bb_params[3] - bb_params[1]  # target_h
    bb[3] = bb_params[2] - bb_params[0]  # target_w
    bb = np.asarray(bb, dtype=np.int32)
    bb = np.reshape(bb, (4,))
    feed_dict_v = {feed_img: test_frames[0][0][np.newaxis,:],
                   feed_att: att_flow[np.newaxis, ..., np.newaxis],
                   feed_prob: seg0_arr[np.newaxis, ..., np.newaxis].astype(np.float32),
                   feed_bb: bb}
    init_lstm_inf_ = sess.run(init_lstm_inf, feed_dict=feed_dict_v)
    current_state_val = np.concatenate((init_lstm_inf_, init_lstm_inf_), axis=0)
    assign_state_ops_ = sess.run(assign_state_ops, feed_dict={feed_state: current_state_val})
    init_seg_mask_, final_seg_mask_, current_state_ = sess.run([init_seg_mask, final_seg_mask, current_state], feed_dict=feed_dict_v)
    current_state_val = np.concatenate((current_state_[0], current_state_[1]), axis=0)
    save_name = str(0).zfill(5) + '.png'
    save_path = params_model['save_result_path'] + '/' + save_name
    seg_mask_0th = np.multiply(np.squeeze(att_flow), np.squeeze(init_seg_mask_))
    imsave(save_path, seg_mask_0th)

    ### get ready for the 1th frame
    flow0 = read_flow(flow_files[0])
    new_seg = transf_slow(seg0_arr, flow0)
    att_flow = binary_dilation(new_seg, structure=struct1, iterations=30).astype(new_seg.dtype)
    att_obj = Image.fromarray(np.squeeze(att_flow.astype(np.uint8)))
    bb_params = att_obj.getbbox()  # [w1,h1,w2,h2]
    bb = [0] * 4
    bb[0] = bb_params[1]  # offset_h
    bb[1] = bb_params[0]  # offset_w
    bb[2] = bb_params[3] - bb_params[1]  # target_h
    bb[3] = bb_params[2] - bb_params[0]  # target_w
    bb = np.asarray(bb, dtype=np.int32)
    bb = np.reshape(bb, (4,))
    att_rgb_overlay_path = params_model['save_result_path'].replace('flow_att_seg', 'overlaid/flow-to-att_rgb')
    init_seg_overlay_path = att_rgb_overlay_path.replace('flow-to-att_rgb', 'flow-to-att-to-seg_rgb')
    ###
    for test_idx in range(len(test_frames)):
        print("Inference on frame {}".format(test_idx+1))
        ### Save attention area on current image
        rgb_obj = Image.open(img_files[test_idx+1])
        att_rgb = overlay_seg_rgb(att_flow, rgb_obj)
        att_rgb_save_name = str(test_idx+1).zfill(5) + '.png'
        att_rgb_save_path = att_rgb_overlay_path + '/' + att_rgb_save_name
        att_rgb.save(att_rgb_save_path)
        ### 1st forward pass, only get init seg estimation
        feed_dict_v = {feed_img: test_frames[test_idx+1][0][np.newaxis,:],
                       feed_att: att_flow[np.newaxis, ..., np.newaxis]}
        init_prob_map_, init_seg_mask_ = sess.run([init_prob_map, init_seg_mask], feed_dict=feed_dict_v) # [1,H,W]
        save_name = str(test_idx+1).zfill(5) + '.png'
        save_path = params_model['save_result_path'] + '/' + save_name
        prob_path = params_model['save_prob_path'] + '/' + save_name
        # save init seg mask
        init_seg_mask_ = np.multiply(np.squeeze(att_flow), np.squeeze(init_seg_mask_))
        init_prob_map_ = np.multiply(np.squeeze(att_flow.astype(np.float32)), np.squeeze(init_prob_map_))
        # imsave(save_path, init_seg_mask_)
        # imsave(prob_path, init_prob_map_)
        ### Save init seg overlay rgb
        save_init_seg_rgb_path = init_seg_overlay_path + '/' + save_name
        init_seg_rgb = overlay_seg_rgb(init_seg_mask_, rgb_obj)
        init_seg_rgb.save(save_init_seg_rgb_path)
        ### 2nd run, assign lstm state
        assign_state_ops_ = sess.run(assign_state_ops, feed_dict={feed_state: current_state_val})
        ### 3rd run, get final seg and update hidden state
        feed_dict_v = {feed_img: test_frames[test_idx+1][0][np.newaxis,:],
                       feed_att: att_flow[np.newaxis, ..., np.newaxis],
                       feed_prob: init_prob_map_[np.newaxis, ..., np.newaxis],
                       feed_bb: bb}
        final_seg_mask_, current_state_ = sess.run([final_seg_mask, current_state], feed_dict=feed_dict_v)
        # current_state_ tuple ([1,256,512,64],[1,256,512,64])
        current_state_val = np.concatenate((current_state_[0], current_state_[1]), axis=0)
        ### save final seg mask
        final_seg_mask_ = np.multiply(np.squeeze(att_flow), np.squeeze(final_seg_mask_))
        save_path = params_model['save_result_path'] + '/' + save_name
        imsave(save_path, final_seg_mask_)
        ### save final seg overlay rgb
        save_final_seg_rgb_path = init_seg_overlay_path + '/final/' + save_name
        final_seg_rgb = overlay_seg_rgb(final_seg_mask_, rgb_obj)
        final_seg_rgb.save(save_final_seg_rgb_path)
        ### Get next attention
        if test_idx != len(test_frames) - 1:
            flow_arr = read_flow(flow_files[test_idx + 1])
            new_seg = transf_slow(final_seg_mask_, flow_arr)
            att_flow = binary_dilation(new_seg, structure=struct1, iterations=40).astype(new_seg.dtype)
        ### Get next att box
        att_obj = Image.fromarray(np.squeeze(att_flow.astype(np.uint8)))
        bb_params = att_obj.getbbox()  # [w1,h1,w2,h2]
        bb = [0] * 4
        bb[0] = bb_params[1]  # offset_h
        bb[1] = bb_params[0]  # offset_w
        bb[2] = bb_params[3] - bb_params[1]  # target_h
        bb[3] = bb_params[2] - bb_params[0]  # target_w
        bb = np.asarray(bb, dtype=np.int32)
        bb = np.reshape(bb, (4,))
        print("Saved result.")
        print("Finished inference.")









