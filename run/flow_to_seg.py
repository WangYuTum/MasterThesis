from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import glob
from PIL import Image
import math
from scipy.misc import imsave
TAG_FLOAT = 202021.25


def get_val_seq_paths():

    seq_num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 15, 18, 20, 23, 24, 25, 27, 29]
    val_seq_txt = '../../../DAVIS17_train_val/ImageSets/2017/val.txt'
    tmp_seq_paths = []
    with open(val_seq_txt) as t:
        val_seq_names = t.read().splitlines()  # e.g. [bike-packing, blackswan, ...]
    for i in range(len(val_seq_names)):
        val_seq_path = os.path.join('../../../DAVIS17_train_val/JPEGImages/480p', val_seq_names[i])
        tmp_seq_paths.append(val_seq_path)
    val_seq_paths = []
    for idx in seq_num_list:
        val_seq_paths.append(tmp_seq_paths[idx])
    print('Got {} val seqs.'.format(len(val_seq_paths)))

    return val_seq_paths


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


def get_pair_data(seq_path):
    '''
    Get rgb/seg/flow_arr/flow_color data for the seq.

    :param seq_path: a seq path, string
    :return: a list [(rgb0, seg0, flow0), (rgb1, seg1, flow1), ...]
    '''
    rgb_search = os.path.join(seq_path, '*.jpg')
    rgb_files = glob.glob(rgb_search)
    rgb_files.sort()
    seg_search = os.path.join(seq_path.replace('JPEGImages', 'Annotations'), '*.png')
    seg_files = glob.glob(seg_search)
    seg_files.sort()
    flow_search = os.path.join(seq_path.replace('JPEGImages', 'Flow'), '*.flo')
    flow_files = glob.glob(flow_search)
    flow_files.sort()
    flow_color_search = os.path.join(seq_path.replace('JPEGImages', 'FlowColor'), '*.png')
    flow_color_files = glob.glob(flow_color_search)
    flow_color_files.sort()
    num_rgb = len(rgb_files)
    assert num_rgb == len(seg_files) and num_rgb == len(flow_files)+1, 'Number of rgb/seg/flow does not match!'
    assert len(flow_files) == len(flow_color_files), 'Number of flow_arr/flow_png does not match!'

    pair_data = []
    for i in range(num_rgb):
        img_obj = Image.open(rgb_files[i])
        seg_arr = np.greater(np.array(Image.open(seg_files[i]), np.uint8), 0).astype(np.uint8)
        h = img_obj.size[1]
        w = img_obj.size[0]
        if i != num_rgb-1:
            flow_arr = read_flow(flow_files[i])
            flow_color = Image.open(flow_color_files[i])
        else:
            flow_arr = np.zeros((h,w,2), np.float32)
            flow_color = Image.fromarray(np.zeros((h,w), np.uint8), mode='P')
        assert h == seg_arr.shape[0] and h == flow_arr.shape[0] and h == flow_color.size[1], 'Shape error!'
        assert w == seg_arr.shape[1] and w == flow_arr.shape[1] and w == flow_color.size[0], 'Shape error!'
        pair_data.append((img_obj, seg_arr, flow_arr, flow_color))
    print('Got {} triple pairs for seq {}'.format(len(pair_data), seq_path.split('/')[-1]))

    return pair_data


def primary_func(pair_data, flow_to_seg_rgb_root, flow_rgb_root, flow_to_seg_root):

    # Input format: [(rgb0, seg0, flow_arr0, flow_color0), (), ...]
    # rgb: [w,h,3], Image obj
    # seg0: [h,w], np.uint8
    # flow_arr: [h,w,2], np.float32; the last flow_arr is zeros
    # flow_color: [h,w], Image obj; the last flow_color is zeros
    num_pairs = len(pair_data)
    current_seg = pair_data[0][1]
    for idx, pair in enumerate(pair_data):
        img_obj = pair[0]
        flow_arr = pair[2]
        flow_color = pair[3]
        seg_rgb_blend = overlay_seg_rgb(current_seg, img_obj)
        seg_rgb_blend.save(flow_to_seg_rgb_root + '/' + str(idx).zfill(5) + '.png')
        if idx != num_pairs - 1:
            flow_rgb_blend = overlay_flow_rgb(img_obj, flow_color) # Image obj
            flow_rgb_blend.save(flow_rgb_root + '/' + str(idx).zfill(5) + '.png')
            new_seg = generate_seg(current_seg, flow_arr) # np.array, [h,w], uint8
            new_seg_obj = Image.fromarray(new_seg, mode='P')
            new_seg_obj.putpalette([0,0,0,255,255,255])
            new_seg_obj.save(flow_to_seg_root + '/' + str(idx+1).zfill(5) + '.png')
            current_seg = new_seg


def generate_seg(seg_arr, flow_arr):

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
                # if flow_arr[idx_h][idx_w][0] >= 0:
                #     motion_h = int(math.ceil(flow_arr[idx_h][idx_w][0]))
                # else:
                #     motion_h = int(math.floor(flow_arr[idx_h][idx_w][0]))
                # if flow_arr[idx_h][idx_w][1] >= 0:
                #     motion_w = int(math.ceil(flow_arr[idx_h][idx_w][1]))
                # else:
                #     motion_w = int(math.floor(flow_arr[idx_h][idx_w][1]))
                motion_h = int(round(flow_arr[idx_h][idx_w][1]))
                motion_w = int(round(flow_arr[idx_h][idx_w][0]))
                new_h = idx_h + motion_h
                new_w = idx_w + motion_w
                if new_h < h and new_h >= 0 and new_w < w and new_w >= 0:
                    new_seg[new_h][new_w] = 1

    return new_seg


def overlay_flow_rgb(rgb_obj, flow_obj):

    com_flow_img = Image.blend(rgb_obj, flow_obj, 0.6)

    return com_flow_img


def overlay_seg_rgb(seg_arr, rgb_obj):

    img_palette = np.array([0, 0, 0, 150, 0, 0])
    seg_color = Image.fromarray(seg_arr, mode='P')
    seg_color.putpalette(img_palette)
    overlay_mask = Image.fromarray(seg_arr * 150, mode='L')

    com_img = Image.composite(seg_color, rgb_obj, overlay_mask)

    return com_img


def main():

    seq_paths = get_val_seq_paths()
    num_seq = len(seq_paths)
    for seq_idx in range(num_seq):
        print('Processing seq {}...'.format(seq_paths[seq_idx].split('/')[-1]))
        pair_data = get_pair_data(seq_paths[seq_idx])
        flow_to_seg_root = '../data/results/flow-to-seg/' + seq_paths[seq_idx].split('/')[-1]
        flow_rgb_root = '../data/results/overlaid/flow_rgb/' + seq_paths[seq_idx].split('/')[-1]
        flow_to_seg_rgb_root = '../data/results/overlaid/flow-to-seg_rgb/' + seq_paths[seq_idx].split('/')[-1]
        primary_func(pair_data, flow_to_seg_rgb_root, flow_rgb_root, flow_to_seg_root)
    print('Job done.')


if __name__ == '__main__':
    main()