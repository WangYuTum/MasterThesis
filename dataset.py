'''
    The data pipeline for DAVIS single object attention network.

    Goal: loading all frames into memory and randomly choose one sequence during training

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

    TODO
    Pre-processing(to be done in train script):
        - Convert np.uint8 to proper datatype(np.float32 for imgs, np.int32 for gts)
        - Randomly resize img/gt pairs
        - Randomly left/right flips
        - Subtract mean and divide var for images
        - Dilate gts to get attention map
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
from PIL import Image
import sys
import os
import glob
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imread

class DAVIS_dataset():
    def __init__(self, params):
        '''
            The init method load all sequences' frames into memory.
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._seq_set = params.get('seq_set', None) # e.g. '../../../DAVIS17_train_val/ImageSets/2017/train.txt'

        # some statistics about the data
        self._max_train_len = 100
        self._min_train_len = 25
        self._avg_train_len = 70
        self._max_val_len = 104
        self._min_val_len = 34
        self._avg_val_len = 67
        self._data_mean = np.array([115.195829334, 114.927476686, 107.725750308]).reshape((1,1,3))
        self._data_std = np.array([64.5572961827, 63.0172054007, 67.0494050908]).reshape((1,1,3))

        if self._seq_set is None:
            sys.exit('Must specify a set in txt format.')
        if self._mode is None:
            sys.exit('Must specify a mode.')
        elif self._mode == 'train':
            self._seq_paths = self._load_seq_path()
            self._train_imgs, self._train_gts = self._get_train_data()
            # check length
            if len(self._train_imgs) != len(self._train_gts) or len(self._train_imgs) == 0:
                sys.exit('Train imgs/gts length do not match.')
        elif self._mode == 'val':
            self._seq_paths = self._load_seq_path()
            self._val_imgs, self._val_gts = self._get_val_data()
            if len(self._val_imgs) != len(self._val_gts) or len(self._val_imgs) == 0:
                sys.exit('Val imgs/gts length do not match.')
        else:
            sys.exit('Not supported mode.')

        self._permut_range = len(self._seq_paths)
        print('Data loaded.')

    def _load_seq_path(self):

        seq_paths = []
        with open(self._seq_set) as t:
            seq_names = t.read().splitlines()  # e.g. [bike-packing, blackswan, ...]
        for i in range(len(seq_names)):
            seq_path = os.path.join('../../DAVIS17_train_val/JPEGImages/480p', seq_names[i])
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

    def _get_random_seq_idx(self):

        rand_seq_idx = np.random.permutation(self._permut_range)[0]

        return rand_seq_idx

    def get_random_seq(self):

        rand_seq_idx = self._get_random_seq_idx()
        seq_imgs = self._train_imgs[rand_seq_idx] # list: [img0, img1, ...]
        seq_gts = self._train_gts[rand_seq_idx] # list: [gt0, gt1, ...]

        return seq_imgs, seq_gts

    def get_one_shot_pair(self):
        '''
        :return: [img, gt, weight] for the one-shot fine-tuning of the self._seq
        '''
        pair = []
        pair.append(self._seq[0])

        gt_path = os.path.join(self._seq_path.replace('JPEGImages','Annotations'), '00000.png')
        gt = np.array(Image.open(gt_path), dtype=np.int8)
        # convert to binary
        gt_bool = np.greater(gt, 0)
        gt_bin = gt_bool.astype(np.uint8)

        gt = np.expand_dims(gt_bin, axis=-1) # [H,W,1]
        pair.append(gt)

        # Compute balanced weight for training, [H,W,1]
        num_pos = np.sum(gt)
        num_neg = np.sum(1-gt)
        num_total = num_pos + num_neg
        weight_pos = num_neg.astype(np.float32) / num_total.astype(np.float32)
        weight_neg = 1.0 - weight_pos
        mat_pos = np.multiply(gt.astype(np.float32), weight_pos)
        mat_neg = np.multiply((1.0-gt).astype(np.float32), weight_neg)
        mat_weight = np.add(mat_pos, mat_neg)
        pair.append(mat_weight)

        return pair

    def get_test_frames(self):

        frame_list = self._seq[1:]

        return frame_list

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



