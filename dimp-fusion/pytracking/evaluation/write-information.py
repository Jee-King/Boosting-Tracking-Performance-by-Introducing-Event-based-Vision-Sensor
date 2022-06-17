import os

val_path = '/home/iccd/data/zjq-reshot33/annotation'
all_seqs = os.listdir(val_path)

for seq in all_seqs:
    if seq == 'list.txt':
        continue
    frame_num = len(os.listdir(os.path.join(val_path,seq,'img')))
    with open('seq.txt', 'a') as f:
        f.writelines(['{{\'anno_path\': \'{}/groundtruth_rect.txt\','.format(seq) , '\n' , \
                      '\'endFrame\': {},'.format(frame_num), '\n', \
                      '\'ext\': \'jpg\',', '\n', \
                      '\'name\': \'{}\','.format(seq), '\n', \
                      '\'nz\': 4,', '\n', \
                      '\'object_class\': \'object\',', '\n', \
                      '\'path\': \'{}/img\','.format(seq), '\n', \
                      '\'startFrame\': 1},', '\n'])
