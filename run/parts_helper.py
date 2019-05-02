'''
    Methods that are used in tracking and segmenting parts
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import numpy as np
from random import shuffle
from PIL.ImageDraw import Draw

IoU_TH = 0.1

def is_valid_bbox_mask(seg_arr, bbox, mask_path):
    '''
    :param seg_arr: [480, 854], np.uint8
    :param bbox: [xmin, ymin, xmax, ymax]
    :param mask_path: path of part mask
    :return:
    '''
    mask_obj = Image.open(mask_path)
    mask_arr = np.array(mask_obj, np.uint8) # [480, 854]
    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    iou_val = IoU_mask(part_mask=mask_arr, obj_mask=seg_arr, bbox_area=bbox_area)
    if iou_val < IoU_TH:
        return False
    else:
        return True

def is_valid_bbox(seg_arr, bbox):


    zeros_arr = np.zeros((480, 854), np.uint8)  # (480, 854)
    zeros_obj = Image.fromarray(zeros_arr)  # [854, 480]

    draw_handle = Draw(zeros_obj)
    draw_handle.rectangle(xy=bbox, fill=1, outline=0)
    part_mask = np.array(zeros_obj, np.uint8)

    bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
    iou_val = IoU_mask(part_mask=part_mask, obj_mask=seg_arr, bbox_area=bbox_area)
    if iou_val < 0.05:
        return False
    else:
        return True

def IoU_mask(part_mask, obj_mask, bbox_area):
    shape0 = np.shape(part_mask)
    shape1 = np.shape(obj_mask)
    if not np.array_equal(shape0, shape1):
        raise ValueError('Shape of part mask {} != obj mask {}'.format(shape0, shape1))

    # do element-wise multiplication to get intersection
    inters = np.sum(np.multiply(part_mask, obj_mask))
    iou_val = float(inters) / float(bbox_area)

    return iou_val

def draw_mul_bbox_mask(img_arr, boxes, colors, save_dir):
    '''
    :param img_arr: [h, w, 3], np.array
    :param boxes: list of boxes, each is [xmin, ymin, xmax, ymax]
    :param colors: list of possible colors
    :param indices: list of part indices to be drawn
    :param save_dir:
    :return: None
    '''

    num_bbox = len(boxes)
    # blend masks
    blended = img_arr.astype(np.int32)

    # draw boxes
    img_obj = Image.fromarray(blended.astype(np.uint8))
    draw_handle = Draw(img_obj)
    for i in range(num_bbox):
        bbox = boxes[i]
        draw_handle.rectangle(bbox, outline=colors[i])

    # save
    img_obj.save(save_dir)

def gen_box_colors():

    # generate maximum 448 colors
    r_colors = [24, 48, 72, 96, 120, 144, 168, 192] # r_num = 8
    g_colors = [24, 48, 72, 96, 120, 144, 168, 192] # g_num = 8
    b_colors = [24, 48, 72, 96, 120, 144, 168, 192] # b_num = 7
    shuffle(r_colors)
    shuffle(g_colors)
    shuffle(b_colors)

    colors = []
    for r_i in r_colors:
        for g_i in g_colors:
            for b_i in b_colors:
                color = (r_i, g_i, b_i)
                colors.append(color)

    return colors