from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os,sys
import numpy as np
import tensorflow as tf
os.chdir('/usr/stud/wangyu/PycharmProjects/MasterThesis/run')

sys.path.append("..")
from run.parts_helper import is_valid_bbox_mask, draw_mul_bbox_mask, gen_box_colors, is_valid_bbox
from dataset import DAVIS_dataset
from core import resnet
from scipy.misc import imsave
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
import glob
from PIL import Image
TAG_FLOAT = 202021.25

"""
def read_flow(file):

    print('load flow file {}'.format(file))
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
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
"""
def overlay_seg_rgb(seg_arr, rgb_obj):

    seg_arr = seg_arr.astype(np.uint8)
    img_palette = np.array([0, 0, 0, 150, 0, 0])
    seg_color = Image.fromarray(seg_arr, mode='P')
    seg_color.putpalette(img_palette)
    overlay_mask = Image.fromarray(seg_arr * 150, mode='L')

    com_img = Image.composite(seg_color, rgb_obj, overlay_mask)

    return com_img

arg_seq_name = str(sys.argv[1])
arg_frame_id = int(sys.argv[2])
#arg_seq_name = 'blackswan'
#arg_frame_id = 3

# set fine-tune or test
seq_path = os.path.join('../../../DAVIS17_train_val/JPEGImages/480p', arg_seq_name)

# config device
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
config_gpu = tf.ConfigProto()
config_gpu.gpu_options.allow_growth = True

# get all val seq paths
val_seq_txt = '../../../DAVIS17_train_val/ImageSets/2017/val.txt'
val_seq_paths = []
with open(val_seq_txt) as t:
    val_seq_names = t.read().splitlines() # e.g. [bike-packing, blackswan, ...]
for i in range(len(val_seq_names)):
    val_seq_path = os.path.join('../../../DAVIS17_train_val/JPEGImages/480p', val_seq_names[i])
    val_seq_paths.append(val_seq_path)

# config dataset params (480p images)
params_data = {
    'mode': 'val',
    'seq_path': seq_path,
}

# get seq data
with tf.device('/cpu:0'):
    val_data = DAVIS_dataset(params_data)
    test_frames = val_data.get_test_frames() # list: [[img1, att1], [img2, att2], ...], img with [H,W,3], att [H,W,1]
print('Load test data for seq {} done.'.format(seq_path.split('/')[-1]))
feed_img = tf.placeholder(tf.float32, [1, None, None, 3])
feed_att = tf.placeholder(tf.int32, [1, None, None, 1])

params_model = {
    'batch': 1,
    'data_format': 'NCHW',  # optimal for cudnn
    'restore_fine-tune_bin': '/work/wangyu/CNN-part-gate-img-v4_large/40ep/'+seq_path.split('/')[-1]+'/fine-tune.ckpt-24300',
    'save_result_path': '../data/results/flow_att_seg/'+seq_path.split('/')[-1],
    'save_prob_path': '../data/results/prob_map/'+seq_path.split('/')[-1]
}

# build network, on GPU by default
model = resnet.ResNet(params_model)
prob_map, mask = model.test(feed_img, feed_att) # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
init_op = tf.global_variables_initializer()
sum_all = tf.summary.merge_all()
# define Saver
saver_test = tf.train.Saver()

# run session
with tf.Session(config=config_gpu) as sess:

    sess.run(init_op)
    # restore all variables
    saver_test.restore(sess, params_model['restore_fine-tune_bin'])
    print('restored variables from {}'.format(params_model['restore_fine-tune_bin']))
    # set deconv filters
    print('All weights initialized.')

    # starting testing
    print("Starting inference for {}".format(seq_path.split('/')[-1]))
    seg0_file = os.path.join(seq_path.replace('JPEGImages', 'Annotations'), '00000.png')
    seg0_arr = np.greater(np.array(Image.open(seg0_file)), 0).astype(np.uint8)
    struct1 = generate_binary_structure(2, 2)

    ### get attention from siamese
    load_att_mask = '/storage/slurm/wangyu/davis16/results_parts_assemble/'+arg_seq_name+'/'+str(arg_frame_id).zfill(5) + '.png'
    print('load pre_att mask: {}'.format(load_att_mask))
    att_mask = np.array(Image.open(load_att_mask))
    original_att_mask = att_mask
    att_mask = binary_dilation(att_mask, structure=struct1, iterations=20).astype(att_mask.dtype)

    ###
    ### Get image lists of the current sequence
    img_search = os.path.join(seq_path, '*.jpg')
    img_files = glob.glob(img_search)
    img_files.sort()
    att_rgb_overlay_path = params_model['save_result_path'].replace('flow_att_seg', 'overlaid/flow-to-att_rgb')
    seg_overlay_path = att_rgb_overlay_path.replace('flow-to-att_rgb', 'flow-to-att-to-seg_rgb')
    ###
    #for test_idx in range(len(test_frames)):
    print("Inference on frame {}".format(arg_frame_id))
    ### Save attention area on current image
    print('load rgb frame: {}'.format(img_files[arg_frame_id]))
    rgb_obj = Image.open(img_files[arg_frame_id])
    att_rgb = overlay_seg_rgb(att_mask, rgb_obj)
    att_rgb_save_name = str(arg_frame_id).zfill(5) + '.png'
    att_rgb_save_path = att_rgb_overlay_path + '/' + att_rgb_save_name
    att_rgb.save(att_rgb_save_path)
    ###
    feed_dict_v = {feed_img: test_frames[arg_frame_id-1][0][np.newaxis,:],
                   feed_att: att_mask[np.newaxis,...,np.newaxis]}
    # prob_map: [1,H,W] tf.float32, mask: [1,H,W] tf.int16
    prob_map_, mask_ = sess.run([prob_map, mask], feed_dict=feed_dict_v)
    save_name = str(arg_frame_id).zfill(5) + '.png'
    save_path = params_model['save_result_path'] + '/' + save_name
    prob_path = params_model['save_prob_path'] + '/' + save_name
    # save binary mask
    final_seg = np.multiply(np.squeeze(att_mask), np.squeeze(mask_))
    final_prob = np.multiply(np.squeeze(att_mask), np.squeeze(prob_map_))
    imsave(save_path, final_seg)
    imsave(prob_path, final_prob)
    ### Save seg overlay rgb
    save_seg_rgb_path = seg_overlay_path + '/' + save_name
    seg_rgb = overlay_seg_rgb(final_seg, rgb_obj)
    seg_rgb.save(save_seg_rgb_path)
    ###

    print("Saved result.")
    print('Filter bbox/masks ...')
    ### filter bad bbox/masks
    result_base = '/storage/slurm/wangyu/davis16'
    valid_path = os.path.join(result_base, 'valid_indices', arg_seq_name, str(arg_frame_id).zfill(5) + '.npy')
    pre_bbox_path = os.path.join(result_base, 'pre_bboxes', arg_seq_name, str(arg_frame_id).zfill(5) + '.npy')
    result_parts = os.path.join(result_base, 'results_parts')
    frame_parts_dir = os.path.join(result_parts, arg_seq_name, str(arg_frame_id).zfill(5))
    part_list = sorted([os.path.join(frame_parts_dir, name) for name in os.listdir(frame_parts_dir)]) # list of length num_init

    valid_indices = np.load(valid_path).tolist() # [0, 1, 4, 7, ...]
    pre_bboxes_arr = np.load(pre_bbox_path) # [num_init, 4]
    pre_bboxes_list = []
    for i in range(np.shape(pre_bboxes_arr)[0]):
        pre_bboxes_list.append(pre_bboxes_arr[i].tolist())
    #if len(part_list) != np.shape(pre_bboxes_arr)[0]:
    #    raise ValueError('part list {} != num_bbox {}'.format(len(part_list), np.shape(pre_bboxes_arr)[0]))

    new_valid_indices = []
    sum_seg = np.sum(np.multiply(original_att_mask.astype(np.int32), final_seg.astype(np.int32)))
    sum_all = np.sum(original_att_mask.astype(np.int32))
    seg_att_ratio = float(sum_seg) / float(sum_all)
    if seg_att_ratio < 0.1:
        print('do not filter bboxes')
        new_valid_indices = valid_indices
        valid_indices_arr = np.array(new_valid_indices)
    else:
        print('Number of valid bboxes before: {}'.format(len(valid_indices)))
        for i in range(np.shape(pre_bboxes_arr)[0]):
            # only check those are already valid bbox_mask
            if i in valid_indices:
                box = pre_bboxes_list[i]
                #mask_path = part_list[i]
                #bool_val = is_valid_bbox_mask(seg_arr=final_seg.astype(np.uint8), bbox=box, mask_path=mask_path)
                bool_val = is_valid_bbox(seg_arr=final_seg.astype(np.uint8), bbox=box)
                # if the bbox_mask is valid
                if bool_val:
                    new_valid_indices.append(i)
        valid_indices_arr = np.array(new_valid_indices)
    print('Number of valid bboxes after: {}'.format(len(new_valid_indices)))
    np.save(os.path.join(result_base, 'valid_indices', arg_seq_name, str(arg_frame_id).zfill(5) + '.npy'),
            valid_indices_arr)


    ### plot all filtered bboxes
    result_parts_color = os.path.join(result_base, 'results_parts_color', arg_seq_name, 'all', 'after')
    if not os.path.exists(result_parts_color):
        os.mkdir(result_parts_color)
    box_colors = gen_box_colors()
    plot_bbox_list = []
    for i in range(np.shape(pre_bboxes_arr)[0]):
        if i in new_valid_indices:
            plot_bbox_list.append(pre_bboxes_list[i])
    draw_mul_bbox_mask(img_arr=np.array(rgb_obj),
                       boxes=plot_bbox_list,
                       colors=box_colors,
                       save_dir=os.path.join(result_parts_color, str(arg_frame_id).zfill(5) + '.jpg'))
    print("Finished inference.")

    ### Get next attention
    #if test_idx != len(test_frames) - 1:
    #    att_mask = np.array(Image.open('/storage/slurm/wangyu/davis16/results_parts_assemble/' + arg_seq_name+'/' + str(test_idx+2).zfill(5) + '.png'))
    #    att_mask = binary_dilation(att_mask, structure=struct1, iterations=15).astype(att_mask.dtype)
    ###
    # save prob map
    # imsave(save_path.replace('.png', '_prob.png'), np.squeeze(prob_map_))











