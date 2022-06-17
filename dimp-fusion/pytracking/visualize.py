import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import sys
import matplotlib.pyplot as plt

data_path = '/data/zjq/img_ext'
output_result = '/home/zjq/Documents/pytracking/pytracking/tracking_results'
tracker_name = 'atom'
test_case = 'default'
out_path = os.path.join(output_result, tracker_name, test_case)

len_path = len(os.listdir(data_path)) - len(
    [i for i in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, i))])

black_list = ["bottle_sim"]
white_list = ['bear_motion']
result_path = "/home/zjq/result2"
if not os.path.exists(result_path):
    os.mkdir(result_path)

# for index, filename in enumerate(os.listdir(data_path), 1):
#     if os.path.isdir(os.path.join(data_path, filename)):
for index, file_name in enumerate(os.listdir(data_path), 1):
#     if os.path.isdir(os.path.join(data_path, filename)):
# for file_name in white_list:
# print("Rendering {} {}/{}".format(file_name, index, len_path))
    i = 0
    bb_gt = np.rint(np.loadtxt(os.path.join(data_path, file_name, 'groundtruth_rect.txt'), delimiter=','))
    bb_pre = np.rint(np.loadtxt(os.path.join(out_path, file_name + '.txt'), delimiter='\t'))
    root = os.path.join(data_path, file_name, 'img')
    if not os.path.exists(os.path.join(result_path, file_name)):
        os.mkdir(os.path.join(result_path, file_name))
    for f in sorted(os.listdir(os.path.join(data_path, file_name, 'img'))):
        bb_gti = bb_gt[i]
        bb_prei = bb_pre[i]
        img_path = os.path.join(root, f)
        img = cv2.imread(img_path)
        img = cv2.rectangle(img, (int(bb_gti[0]), int(bb_gti[1])),
                            (int(bb_gti[2] + bb_gti[0]), int(bb_gti[3] + bb_gti[1])),
                            (0, 0, 255), 1)
        img = cv2.rectangle(img, (int(bb_prei[0]), int(bb_prei[1])),
                            (int(bb_prei[2] + bb_prei[0]), int(bb_prei[3] + bb_prei[1])),
                            (255, 0, 255), 1)
        print(os.path.join(result_path, file_name, str(i).zfill(4) + '.jpg'))
        cv2.imwrite(os.path.join(result_path, file_name, str(i).zfill(4) + '.jpg'), img)
        i += 1
