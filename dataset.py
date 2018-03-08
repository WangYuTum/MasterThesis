'''
    The data pipeline for DAVIS encapsulated as a class object.

    NOTE: the code is still under developing.
    TODO:
        * Pipeline for val/test fine-tune/inference
'''

# NOTE: The .tfrecord files are located on /work/wangyu/ on hpccremers4

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from scipy.misc import imsave
import sys
import os
import glob
from scipy.misc import imsave
from scipy.misc import toimage
from scipy.misc import imread

class DAVIS_dataset():
    def __init__(self, params):
        '''
        :param
            mode: 'parent_train_binary'
            batch:  2
            tfrecord: None
        '''

        # Init params
        self._mode = params.get('mode', None)
        self._batch = params.get('batch', 0)
        self._tfrecord = params.get('tfrecord', None)
        self._dataset = None

        # params check
        if self._mode is None:
            sys.exit("Must specify a mode.")
        if self._batch == 0:
            sys.exit("Must specify a batch size.")
        if self._tfrecord is None:
            sys.exit("No valid .tfrecord file nor sequence list.")

        # Build up the pipeline
        if self._mode == 'parent_train_binary':
            self._dataset = self._build_pipeline()
        if self._dataset is None:
            sys.exit("Data pipeline not built.")

    def _build_pipeline(self):

        return None


