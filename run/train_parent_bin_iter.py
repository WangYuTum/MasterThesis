from subprocess import call
import os

num_train_iters = 20 # actual alter_iters is 21 since the 1st iter was done manually
train_flag = 0
epochs = 2
lr = 1e-5
save_ckpt_interval = 1200
now_suffix = 3000 # start from 1200(2ep) + 3000(5ep)
l2_decay = 0.0005
restore_ckpt = "att_bin.ckpt-" + str(now_suffix)
for iter_id in range(num_train_iters):
    restore_ckpt = "att_bin.ckpt-" + str(now_suffix)
    if iter_id % 2 == 0: # train main
        print('Train main branch ...')
        train_flag = 0
        epochs = 2
        lr = 5e-6
        save_ckpt_interval = 1200
        l2_decay = 0.0002
        now_suffix += 1200
        tsboard_subdir = 'iter_'+str(iter_id+1)+'_main'
    else:
        print('Train OF/Feat_trans branch ...')
        train_flag = 1
        epochs = 5
        lr = 5e-5
        save_ckpt_interval = 3000
        l2_decay = 0.0002
        now_suffix += 3000
        tsboard_subdir = 'iter_' + str(iter_id + 1) + '_OF'
    tsboard_dir = '../data/tsboard_logs/attention_bin/CNN-part-gate-img-v4_large_Flowside/'+tsboard_subdir
    if not os.path.exists(tsboard_dir):
        os.mkdir(tsboard_dir)
    arg1 = str(train_flag)
    arg2 = str(epochs)
    arg3 = str(lr)
    arg4 = str(save_ckpt_interval)
    arg5 = str(restore_ckpt)
    arg6 = str(l2_decay)
    arg7 = str(tsboard_subdir)
    call(['python', 'train_parent_bin.py', arg1, arg2, arg3, arg4, arg5, arg6, arg7])

