'''
    The data pipeline for DAVIS single object attention network.

    Goal: loading all frames into memory and randomly choose one image at a time

    Implementation Note: Video frames stored as a list of list:
        - Images: [seq0, seq1, ...], where seqx = [img0, img1, ...]
        - GTs: [seq0, seq1, ...]
            - where seqx = [gt0, gt1, ...] for train data
            - where seqx = gtx for val data
        - All gts are converted to binary mask
        - Images in np.uint8, GTs in np.uint8 for memory efficiency.
        - All images shape: [H, W, 3], e.g. [480, 854, 3]
          All gts shape: [H, W], e.g. [480, 854]

    About sequence length: Fix the length to max length of all train sequences;
        Shorter sequences pad with previous frames in a reversing order until reaches max length

    Train/val split:
        - Only use train set during training
        - Use val set to do validation

    NOTE:
        - The dataset implementation only include train/val sets.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from PIL import Image
import sys
import os
import glob
from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage import generate_binary_structure
from scipy.misc import imsave
from skimage.transform import resize
from scipy.misc import toimage
from scipy.misc import imread

# data_mean = np.array([115.195829334, 114.927476686, 107.725750308]).reshape((1,1,3))
# data_std = np.array([64.5572961827, 63.0172054007, 67.0494050908]).reshape((1,1,3))
data_mean = np.array([0.485, 0.456, 0.406]).reshape((1,1,3)).astype(np.float32) # the ILSVRC mean, in rgb
data_std = np.array([0.229, 0.224, 0.225]).reshape((1,1,3)).astype(np.float32) # the ILSVRC std, in rgb

class DAVIS_dataset():
    def __init__(self, params):
        '''
            The init method load all sequences' frames into memory.
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._seq_set = params.get('seq_set', None) # e.g. '../../../DAVIS17_train_val/ImageSets/2017/train.txt'
        self._seq_path = params.get('seq_path', None) # used in inference
        self._val_seq_frames = None # used in inference

        # some statistics about the data
        self._max_train_len = 100
        self._min_train_len = 25
        self._avg_train_len = 70
        self._max_val_len = 104
        self._min_val_len = 34
        self._avg_val_len = 67
        # self._data_mean = np.array([115.195829334, 114.927476686, 107.725750308]).reshape((1,1,3))
        # self._data_std = np.array([64.5572961827, 63.0172054007, 67.0494050908]).reshape((1,1,3))

        if self._mode is None:
            sys.exit('Must specify a mode.')
        elif self._mode == 'train':
            if self._seq_set is None:
                sys.exit('Must specify a set in txt format.')
            self._seq_paths = self._load_seq_path()
            self._train_imgs, self._train_gts = self._get_train_data()
            # check length
            if len(self._train_imgs) != len(self._train_gts) or len(self._train_imgs) == 0:
                sys.exit('Train imgs/gts length do not match.')
        elif self._mode == 'val':
            if self._seq_path is None:
                sys.exit('Must specify seq_path in mode {}'.format(self._mode))
            else:
                self._val_seq_frames = self._get_val_frames()
                print('Got {0} frames for seq {1}'.format(len(self._val_seq_frames), self._seq_path))
        else:
            sys.exit('Not supported mode.')

        if self._mode == 'train':
            self._permut_range_seq = len(self._seq_paths)
            self._permut_range_frame = self._max_train_len
        print('Data loaded.')

    def _load_seq_path(self):

        seq_paths = []
        with open(self._seq_set) as t:
            seq_names = t.read().splitlines()  # e.g. [bike-packing, blackswan, ...]
        for i in range(len(seq_names)):
            seq_path = os.path.join('../../../DAVIS17_train_val/JPEGImages/480p', seq_names[i])
            seq_paths.append(seq_path)
        print('Got {} seqs in total.'.format(len(seq_paths)))

        return seq_paths

    def _get_train_data(self):

        train_imgs = []
        train_gts = []
        num_seq = len(self._seq_paths)
        for seq_idx in range(num_seq):
            seq_imgs = []
            seq_gts = []
            search_seq_imgs = os.path.join(self._seq_paths[seq_idx], "*.jpg")
            files_seq = glob.glob(search_seq_imgs)
            files_seq.sort()
            num_frames = len(files_seq)
            print('{} frames for seq {}'.format(num_frames, seq_idx))
            if num_frames == 0:
                sys.exit("Got no frames for seq {}".format(self._seq_paths[seq_idx]))
            for i in range(num_frames):
                frame_img = np.array(Image.open(files_seq[i])).astype(np.uint8)
                frame_gt_path = files_seq[i].replace('JPEGImages', 'Annotations')
                frame_gt_path = frame_gt_path.replace('jpg', 'png')
                frame_gt = np.array(Image.open(frame_gt_path)).astype(np.uint8)
                # convert to binary
                gt_bool = np.greater(frame_gt, 0)
                gt_bin = gt_bool.astype(np.uint8)
                seq_imgs.append(frame_img)
                seq_gts.append(gt_bin)
            # go several rounds to fill up the required length
            max_round = int(self._max_train_len / self._min_train_len) # 100 / 25 = 5 for train seqs
            break_round = 0
            for _ in range(max_round):
                if break_round == 1: # if no further round required, break
                    break
                else:
                    start_base = len(seq_imgs) - 2
                for in_idx in range(num_frames-1):
                    if len(seq_imgs) == self._max_train_len:
                        break_round = 1
                        break
                    else:
                        seq_imgs.append(seq_imgs[start_base-in_idx])
                        seq_gts.append(seq_gts[start_base-in_idx])
            print('After, {} frames for seq {}'.format(len(seq_imgs), seq_idx))
            train_imgs.append(seq_imgs)
            train_gts.append(seq_gts)

        return train_imgs, train_gts

    def get_a_random_sample(self):
        '''
        :return: img [1, h, w, 3] float32, seg [1, h, w, 1] int32, weight [1, h, w, 1] float32
        '''

        # choose an image randomly
        seq_imgs, seq_gts = self.get_random_seq() # [img0, img1, ...], [seq0, seq1, ...], both np.uint8
        rand_frame_idx = np.random.permutation(self._permut_range_frame)[0]
        img = seq_imgs[rand_frame_idx] # [h, w, 3], np.uint8
        seg = seq_gts[rand_frame_idx] # [h,w], np.uint8

        # random resize/flip
        stacked = np.concatenate((img, seg[..., np.newaxis]), axis=-1) # [h, w, 4]
        if get_flip_bool():
            stacked = np.fliplr(stacked)
        img_H = np.shape(img)[0]
        img_W = np.shape(img)[1]
        scale = get_scale()
        new_H = int(img_H * scale)
        new_W = int(img_W * scale)

        img_obj = Image.fromarray(stacked[:, :, 0:3], mode='RGB')
        img_obj = img_obj.resize((new_W, new_H), Image.BILINEAR)
        img = np.array(img_obj, img.dtype) # [h, w, 3], np.uint8

        seg_obj = Image.fromarray(np.squeeze(stacked[:, :, 3:4]), mode='L')
        seg_obj = seg_obj.resize((new_W, new_H), Image.NEAREST)
        seg = np.array(seg_obj, seg.dtype)[..., np.newaxis] # [h, w, 1], np.uint8

        # Gating operation
        struct1 = generate_binary_structure(2, 2)
        att = binary_dilation(np.squeeze(seg), structure=struct1, iterations=30).astype(seg.dtype)

        # standardize
        img = img.astype(np.float32) * 1.0 / 255.0
        img -= data_mean
        img /= data_std
        seg = seg.astype(np.int32) # [h, w, 1], np.int32
        att = att.astype(np.int32)[..., np.newaxis] # [h, w, 1], np.int32

        # get balance weight
        weight = get_seg_balance_weight(seg, att) # [h,w,1], np.float32
        # weight = get_att_balance_weight(seg) # [h,w,1]

        # reshape
        img = img[np.newaxis, ...]
        seg = seg[np.newaxis, ...]
        weight = weight[np.newaxis, ...]
        att = att[np.newaxis, ...]

        return img, seg, weight, att


    def _get_val_data(self):

        val_imgs = []
        val_gts = []
        num_seq = len(self._seq_paths)
        for seq_idx in range(num_seq):
            seq_imgs = []
            search_seq_imgs = os.path.join(self._seq_paths[seq_idx], "*.jpg")
            files_seq = glob.glob(search_seq_imgs)
            files_seq.sort()
            num_frames = len(files_seq)
            print('{} frames for seq {}'.format(num_frames, seq_idx))
            if num_frames == 0:
                sys.exit("Got no frames for seq {}".format(self._seq_paths[seq_idx]))
            for i in range(num_frames):
                frame_img = np.array(Image.open(files_seq[i])).astype(np.uint8)
                seq_imgs.append(frame_img)
            frame_gt_path = files_seq[0].replace('JPEGImages', 'Annotations')
            frame_gt_path = frame_gt_path.replace('jpg', 'png')
            seq_gt = np.array(Image.open(frame_gt_path)).astype(np.uint8)
            # convert to binary
            gt_bool = np.greater(seq_gt, 0)
            gt_bin = gt_bool.astype(np.uint8)
            val_imgs.append(seq_imgs)
            val_gts.append(gt_bin)

        return val_imgs, val_gts

    def _get_val_frames(self):
        '''
        Example self._seq_path: 'DAVIS_root/JPEGImages/480p/bike-packing'
        :return: A list of numpy image arrays
        '''

        seq_frames = []
        search_seq_imgs = os.path.join(self._seq_path, "*.jpg")
        files_seq = glob.glob(search_seq_imgs)
        files_seq.sort()
        num_frames = len(files_seq)
        if num_frames == 0:
            sys.exit("Got no frames for seq {}".format(self._seq_path))
        for i in range(num_frames):
            frame = np.array(Image.open(files_seq[i])).astype(np.float32)
            # standardize
            frame = frame.astype(np.float32) * 1.0 / 255.0
            frame -= data_mean
            frame /= data_std
            seq_frames.append(frame)

        return seq_frames

    def _get_random_seq_idx(self):

        rand_seq_idx = np.random.permutation(self._permut_range_seq)[0]

        return rand_seq_idx

    def get_random_seq(self):

        rand_seq_idx = self._get_random_seq_idx()
        seq_imgs = self._train_imgs[rand_seq_idx] # list: [img0, img1, ...]
        seq_gts = self._train_gts[rand_seq_idx] # list: [gt0, gt1, ...]

        return seq_imgs, seq_gts

    def get_one_shot_pair(self):
        '''
        :return: [img, gt, weight] for the one-shot fine-tuning of the self._val_seq_frames
        '''
        pair = []
        pair.append(self._val_seq_frames[0])

        gt_path = os.path.join(self._seq_path.replace('JPEGImages','Annotations'), '00000.png')
        gt = np.array(Image.open(gt_path), dtype=np.int8)
        # convert to binary
        gt_bool = np.greater(gt, 0)
        gt_bin = gt_bool.astype(np.uint8)
        gt = np.expand_dims(gt_bin, axis=-1) # [H,W,1]

        struct1 = generate_binary_structure(2, 2)
        att = binary_dilation(np.squeeze(gt), structure=struct1, iterations=30).astype(gt.dtype)

        gt = gt.astype(np.int32) # [h, w, 1], np.int32
        pair.append(gt)
        att = att.astype(np.int32)[..., np.newaxis] # [h, w, 1], np.int32

        # get balance weight
        weight = get_seg_balance_weight(gt, att) # [h,w,1], np.float32
        pair.append(weight)
        pair.append(att)

        # Compute balanced weight for training, [H,W,1]
        # num_pos = np.sum(gt)
        # num_neg = np.sum(1-gt)
        # num_total = num_pos + num_neg
        # weight_pos = num_neg.astype(np.float32) / num_total.astype(np.float32)
        # weight_neg = 1.0 - weight_pos
        # mat_pos = np.multiply(gt.astype(np.float32), weight_pos)
        # mat_neg = np.multiply((1.0-gt).astype(np.float32), weight_neg)
        # mat_weight = np.add(mat_pos, mat_neg)
        # pair.append(mat_weight)

        return pair

    def get_test_frames(self):

        # Assuming we know the attention area/window
        frame_list = self._val_seq_frames[1:] # shape of frame [h,w,3]
        num_frames = len(frame_list)
        frame_pair = []
        gt_search_path = os.path.join(self._seq_path.replace('JPEGImages', 'Annotations'), '*.png')
        files_gt = glob.glob(gt_search_path)
        files_gt.sort()
        for i in range(num_frames):
            frame_gt = np.array(Image.open(files_gt[i+1])).astype(np.uint8)
            # convert to binary
            gt_bool = np.greater(frame_gt, 0)
            gt_bin = gt_bool.astype(np.uint8) # [h,w], np.uint8
            struct1 = generate_binary_structure(2, 2)
            att = binary_dilation(gt_bin, structure=struct1, iterations=30).astype(gt_bin.dtype)
            att = att.astype(np.int32)[..., np.newaxis]  # [h, w, 1], np.int32
            frame_pair.append([frame_list[i], att])

        return frame_pair

#     Pre-processing(called in train script):
#        (done)- Convert np.uint8 to proper datatype(np.float32 for imgs, np.int32 for gts)
#        (done)- Randomly resize/flip must be done sequence-wise
#        (done)- Subtract mean and divide var for images
#        (done) - Dilate gts to get attention map


def standardize(f0, f1, f2, f3, s0, s1, s2, s3):
    '''
    :param f0, f1, f2, f3: RGB img, shape [H,W,3], np.uint8
    :param s0, s1, s2, s3: Seg_gt, binary mask, shape [H,W,1], np.uint8
    :return: standardized, datatype converted
    '''

    f0 = f0.astype(np.float32)
    f1 = f1.astype(np.float32)
    f2 = f2.astype(np.float32)
    f3 = f3.astype(np.float32)
    s0 = s0.astype(np.int32)
    s1 = s1.astype(np.int32)
    s2 = s2.astype(np.int32)
    s3 = s3.astype(np.int32)

    f0 -= data_mean
    f0 /= data_std
    f1 -= data_mean
    f1 /= data_std
    f2 -= data_mean
    f2 /= data_std
    f3 -= data_mean
    f3 /= data_std

    return f0, f1, f2, f3, s0, s1, s2, s3


def ge_att_pairs(s0, s1, s2, s3):
    '''
    :param s0, s1, s2, s3: Seg_gt for f0, binary mask, shape [H,W], np.uint8
    :return: a0, a1, a2, a3, a01, a23: shape [H,W], np.uint8
    '''

    struct1 = generate_binary_structure(2, 2)
    a0 = binary_dilation(s0, structure=struct1, iterations=30).astype(s0.dtype)
    a1 = binary_dilation(s1, structure=struct1, iterations=30).astype(s1.dtype)
    a2 = binary_dilation(s2, structure=struct1, iterations=30).astype(s2.dtype)
    a3 = binary_dilation(s3, structure=struct1, iterations=30).astype(s3.dtype)
    a01 = binary_dilation(s0, structure=struct1, iterations=40).astype(s0.dtype)
    a23 = binary_dilation(s2, structure=struct1, iterations=40).astype(s2.dtype)

    return a0, a1, a2, a3, a01, a23


def random_resize_flip(f0, f1, f2, f3, s0, s1, s2, s3, flip, scale):
    '''
    NOTE: This function must be applied to every frame for a particular sequence,
    each sequence might (not)flip and has different scale of resize.

    :param f0, f1, f2, f3: RGB img, [H,W,3], np.uint8
    :param s0, s1, s2, s3: Seg_gt, binary, shape [H,W], np.uint8
    :param flip: Boolen, flip or not
    :param scale: np.float32, range: [0.6-1.2], the resize scale
    :return: all resized/fliped together, dtype unchanged, binary shapes to [H,W,1]
    '''

    # stack them, converted to np.float32, [H,W,22]
    stacked = np.concatenate((f0, f1, f2, f3,
                              s0[..., np.newaxis], s1[..., np.newaxis],
                              s2[..., np.newaxis], s3[..., np.newaxis]), axis=-1)
    if flip:
        stacked = np.fliplr(stacked)
    img_H = np.shape(f0)[0]
    img_W = np.shape(f0)[1]
    new_H = int(img_H * scale)
    new_W = int(img_W * scale)

    # PIL.Image.resize preserve range, Image.NEAREST, Image.BILINEAR
    # scipy.misc.imresize rescale to (0,255)

    f0_obj = Image.fromarray(stacked[:,:,0:3], mode='RGB')
    f0_obj.resize((new_H, new_W), Image.BILINEAR)
    f0 = np.array(f0_obj, f0.dtype)

    f1_obj = Image.fromarray(stacked[:,:,3:6], mode='RGB')
    f1_obj.resize((new_H, new_W), Image.BILINEAR)
    f1 = np.array(f1_obj, f1.dtype)

    f2_obj = Image.fromarray(stacked[:,:,6:9], mode='RGB')
    f2_obj.resize((new_H, new_W), Image.BILINEAR)
    f2 = np.array(f2_obj, f2.dtype)

    f3_obj = Image.fromarray(stacked[:,:,9:12], mode='RGB')
    f3_obj.resize((new_H, new_W), Image.BILINEAR)
    f3 = np.array(f3_obj, f3.dtype)

    s0_obj = Image.fromarray(np.squeeze(stacked[:,:,12:13]), mode='L')
    s0_obj.resize((new_H, new_W), Image.NEAREST)
    s0 = np.array(s0_obj, s0.dtype)[..., np.newaxis]

    s1_obj = Image.fromarray(np.squeeze(stacked[:,:,13:14]), mode='L')
    s1_obj.resize((new_H, new_W), Image.NEAREST)
    s1 = np.array(s1_obj, s1.dtype)[..., np.newaxis]

    s2_obj = Image.fromarray(np.squeeze(stacked[:,:,14:15]), mode='L')
    s2_obj.resize((new_H, new_W), Image.NEAREST)
    s2 = np.array(s2_obj, s2.dtype)[..., np.newaxis]

    s3_obj = Image.fromarray(np.squeeze(stacked[:,:,15:16]), mode='L')
    s3_obj.resize((new_H, new_W), Image.NEAREST)
    s3 = np.array(s3_obj, s3.dtype)[..., np.newaxis]

    return f0, f1, f2, f3, s0, s1, s2, s3


def get_att_balance_weight(mask):
    '''
    :param mask: [H,W,1], np.int32, binary
    :return: weight matrix, [H,W,1], np.float32
    '''
    num_pos = np.sum(mask)
    num_neg = np.sum(1 - mask)
    num_total = num_pos + num_neg
    weight_pos = num_neg.astype(np.float32) / num_total.astype(np.float32)
    weight_neg = 1.0 - weight_pos
    mat_pos = np.multiply(mask.astype(np.float32), weight_pos)
    mat_neg = np.multiply((1.0 - mask).astype(np.float32), weight_neg)
    mat_weight = np.add(mat_pos, mat_neg)

    return mat_weight


def get_seg_balance_weight(seg, att):
    '''
    :param seg: segmentation gt for f0, [H,W,1], np.int32
    :param att: attention gt for f0, [H,W,1], np.int32
    :return: weight matrix, [H,W,1], np.float32

    NOTE: set all weights to 0s except for attention area,
          compute balance weight within attention area.
    '''

    num_total = np.sum(att)
    num_pos = np.sum(seg)
    num_neg = num_total - num_pos
    weight_pos = num_neg.astype(np.float32) / num_total.astype(np.float32)
    weight_neg = 1.0 - weight_pos
    mat_pos = np.multiply(seg.astype(np.float32), weight_pos)
    mat_neg = np.multiply(np.subtract(att, seg).astype(np.float32), weight_neg)
    mat_weight = np.add(mat_pos, mat_neg)

    return mat_weight


def get_balance_weights(s0, s1, s2, s3):
    '''
    :param s0: segmentation gt for f0, [H,W,1], np.int32
    :param s1: attention gt for a1, [H,W,1], np.int32
    :param s2: segmentation gt for f2, [H,W,1], np.int32
    :param s3: attention gt for a3, [H,W,1], np.int32
    :return: weight matrix for s0, s1, s2, s3; shape/dtype doesn't change
    '''

    w_s0 = get_att_balance_weight(s0)
    w_s1 = get_att_balance_weight(s1)
    w_s2 = get_att_balance_weight(s2)
    w_s3 = get_att_balance_weight(s3)

    return w_s0, w_s1, w_s2, w_s3


def pack_reshape_batch(b0, b1, b2, b3):
    '''
    :param b0, b1, b2, b3: [H,W,C]
    :return: stacked [4,H,W,C]
    '''
    stacked = np.stack((b0, b1, b2, b3), axis=0)

    return stacked


def get_flip_bool():

    rand_f = np.random.rand()
    if rand_f >= 0.5:
        return True
    else:
        return False


def get_scale():

    rand_i = np.random.randint(6,11)
    rand_scale = float(rand_i) / 10.0

    return rand_scale


# For test purpose
def main():

    mydata = DAVIS_dataset({'mode': 'train',
                            'seq_set': '../../DAVIS17_train_val/ImageSets/2017/train.txt'})

    myseq, mygt = mydata.get_random_seq()
    print('loaded random seq.')
    imsave('ex_img.png', myseq[88])
    imsave('ex_gt.png', mygt[88])

if __name__ == '__main__':
    main()



