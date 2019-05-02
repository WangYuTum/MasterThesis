# The script fine-tune and test on all val seqs
import sys
from subprocess import call

# create param list
"""
params = [(1,1), (0,1),
          (1,2), (0,2),
          (1,3), (0,3),
          (1,4), (0,4),
          (1,5), (0,5),
          (1,6), (0,6),
          (1,7), (0,7),
          (1,8), (0,8),
          (1,9), (0,9),
          (1,11), (0,11),
          (1,12), (0,12),
          (1,13), (0,13),
          (1,15), (0,15),
          (1,18), (0,18),
          (1,20), (0,20),
          (1,23), (0,23),
          (1,24), (0,24),
          (1,25), (0,25),
          (1,27), (0,27),
          (1,29), (0,29)]

params = [(0,1),
          (0,2),
          (0,3),
          (0,4),
          (0,5),
          (0,6),
          (0,7),
          (0,8),
          (0,9),
          (0,11),
          (0,12),
          (0,13),
          (0,15),
          (0,18),
          (0,20),
          (0,23),
          (0,24),
          (0,25),
          (0,27),
          (0,29)]

len_all = len(params)
for i in range(len_all):
    arg0 = str(params[i][0])
    arg1 = str(params[i][1])
    call(['python', 'test_parent_bin.py', arg0, arg1])
          
"""

arg0 = 'breakdance' # arg_seq_name:  blackswan
arg1 = str(1) # arg_frame_id
call(['python', 'test_parent_bin.py', arg0, arg1])



