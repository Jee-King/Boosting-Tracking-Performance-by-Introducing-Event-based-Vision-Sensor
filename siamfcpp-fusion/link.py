import shutil
import os
from os.path import join
# train_list = open('/home/zjq/eotb_train_split.txt').read().splitlines()
# train_dir = '/home/zjq/data/img_ext/'
# for file in train_list:
#     seqs = join('/home/fyk/zjq_tpami/data/train',file)
#     os.makedirs(seqs)
#     os.makedirs(join(seqs,'inter3_stack'))
#     os.makedirs(join(seqs,'img'))

#     os.symlink(join(train_dir,file,'groundtruth_rect.txt'), join('/home/fyk/zjq_tpami/data/train/',file,'groundtruth.txt'))
#     for events in os.listdir(join(train_dir,file,'inter3_stack')):
#         os.symlink(join(train_dir,file,'inter3_stack',events), join('/home/fyk/zjq_tpami/data/train/',file,'inter3_stack',events))
#     for events in os.listdir(join(train_dir,file,'img')):
#         os.symlink(join(train_dir,file,'img',events), join('/home/fyk/zjq_tpami/data/train/',file,'img',events))

train_list = open('/home/zjq/eotb_val_split.txt').read().splitlines()
train_dir = '/home/zjq/data/img_ext/'
for file in train_list:
    seqs = join('/home/fyk/zjq_tpami/data/val',file)
    os.makedirs(seqs)
    os.makedirs(join(seqs,'inter3_stack'))
    os.makedirs(join(seqs,'img'))

    os.symlink(join(train_dir,file,'groundtruth_rect.txt'), join('/home/fyk/zjq_tpami/data/val/',file,'groundtruth.txt'))
    for events in os.listdir(join(train_dir,file,'inter3_stack')):
        os.symlink(join(train_dir,file,'inter3_stack',events), join('/home/fyk/zjq_tpami/data/val/',file,'inter3_stack',events))
    for events in os.listdir(join(train_dir,file,'img')):
        os.symlink(join(train_dir,file,'img',events), join('/home/fyk/zjq_tpami/data/val/',file,'img',events))