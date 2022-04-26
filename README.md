# FENetpp

##  Example Code Evaluation 
 
The code is built on [visionml/pytracking](https://github.com/visionml/pytracking)  and tested on Ubuntu 18.04 environment with RTX 3090 GPUs.

##  Our Method 
1. Download our preprocessed [test dataset](https://drive.google.com/drive/folders/1pNY8kahrof9l9zCw7TtXY4RhvJ4GGx37?usp=sharing) of FE240hz. (The whole FE240hz dataset can be downloaded [here](https://zhangjiqing.com/publication/iccv21_fe108_tracking/)).

2. Download the [pretrained model](https://drive.google.com/file/d/1xD-d24TRoMHRAQKIxE7CxMhI2UffSiUG/view?usp=sharing) and put it into ./snapshots/stnet.

3. Change dataset path at line 32 in videoanalyst/engine/tester/tester_impl/eventdata.py. ```data_root="/your_data_path/img_120_split"```

4. run ``` python main/test.py --config experiments/test/fe240/fe240.yaml ``` the predicted bounding boxes are saved in logs/EVENT-Benchmark/. 
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one event frame.

##  Extended [TransT](https://github.com/chenxin-dlut/TransT) with Our Modules
1. ``` cd TransT-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzWxILiiehXLXC6O?e=jqahvi) and put it into ./checkpoints/ltr/transt/transt.

3. Change dataset and model path from line 8 to line 13 in pytracking/evaluation/local.py 

4. ``` cd pytracking ``` and run ``` python run_tracker.py transt transt50 --dataset eotb --sequence val  --epochname  TransT_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one event frame.

Please cite TransT if you find the work useful:
```
@inproceedings{TransT,
title={Transformer Tracking},
author={Chen, Xin and Yan, Bin and Zhu, Jiawen and Wang, Dong and Yang, Xiaoyun and Lu, Huchuan},
booktitle={CVPR},
year={2021}
}
```
****

##  Extended [Stark](https://github.com/researchmm/Stark) with Our Modules
1. ``` cd Stark-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzTCxJqUH2Zuk0vk?e=IKH9UV) and put it into ./lib/train/checkpoints/train/stark_s/baseline.

3. Change model path and other settings from line 13 to line 19 in ./lib/test/evaluation/local.py 

4. Change project path and data path from line 13 to line 15 in ./lib/test/evaluation/enviorment.py 

4. ``` cd ./tracking ``` and run ``` python test.py stark_s baseline --dataset eotb --sequence val --epochname STARKS_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height] in one event frame.

Please cite Stark if you find the work useful:
```
@inproceedings{yan2021learning,
  title={Learning spatio-temporal transformer for visual tracking},
  author={Yan, Bin and Peng, Houwen and Fu, Jianlong and Wang, Dong and Lu, Huchuan},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={10448--10457},
  year={2021}
}
```
****

##  Acknowledgement

