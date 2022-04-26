# FENetpp

##  Example Code Evaluation 
 
The code is built on [visionml/pytracking](https://github.com/visionml/pytracking)  and tested on Ubuntu 18.04 environment with RTX 3090 GPUs.

##  Our Method 
1. ``` cd FENetpp ```

2. Download our proposed [dataset](https://zhangjiqing.com/dataset/)

3. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzbdrUqGgZasRwTi), and put it into ``` ./pytracking/networks ```

4. Change your own path in ``` ./pytracking/evaluation/local.py ```

5. run ``` python run_tracker.py dimp prdimp18 --dataset eotb --sequence val --epochname fenetpp.pth.tar ```, the predicted bbox will be saved in ``` ./pytracking/tracking_results ```. Using jupyter in ```notebooks``` to see the SR and PR scores.
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

##  Extended [TransT](https://github.com/chenxin-dlut/TransT) with Our Modules
1. ``` cd TransT-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzWxILiiehXLXC6O?e=jqahvi) and put it into ./checkpoints/ltr/transt/transt.

3. Change dataset and model path from line 8 to line 13 in pytracking/evaluation/local.py 

4. ``` cd pytracking ``` and run ``` python run_tracker.py transt transt50 --dataset eotb --sequence val  --epochname  TransT_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

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
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

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
We would like to thank [PyTracking](https://github.com/visionml/pytracking),  [TransT](https://github.com/chenxin-dlut/TransT) and [Stark](https://github.com/researchmm/Stark) for providing great frameworks and toolkits.

