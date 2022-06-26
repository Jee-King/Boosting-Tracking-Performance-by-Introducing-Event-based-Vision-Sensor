# FENetpp

<!-- ##  Example Code Evaluation 
 
The code is built on [visionml/pytracking](https://github.com/visionml/pytracking)  and tested on Ubuntu 18.04 environment with RTX 3090 GPUs. -->

## Download DATASET
Download our proposed [dataset](https://zhangjiqing.com/dataset/)



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

5. ``` cd ./tracking ``` and run ``` python test.py stark_s baseline --dataset eotb --sequence val --epochname STARKS_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
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

##  Extended [PrDiMP](https://github.com/visionml/pytracking) with Our Modules
1. ``` cd fenetpp ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzbdrUqGgZasRwTi), and put it into ``` ./pytracking/networks ```

3. Change your own path in ``` ./pytracking/evaluation/local.py ```

4. run ``` python run_tracker.py dimp prdimp18 --dataset eotb --sequence val --epochname fenetpp.pth.tar ```, the predicted bbox will be saved in ``` ./pytracking/tracking_results ```. Using jupyter in ```notebooks``` to see the SR and PR scores.
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite PrDiMP if you find the work useful:
```
@inproceedings{danelljan2020probabilistic,
  title={Probabilistic regression for visual tracking},
  author={Danelljan, Martin and Gool, Luc Van and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={7183--7192},
  year={2020}
}
```
****

##  Extended [ATOM](https://github.com/visionml/pytracking) with Our Modules
1. ``` cd atom-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xooUoiI8aAdnhmVDDf) and put it into $YOUR_PATH.

3. Change model path and other settings from line 9 to line 20 in ./lib/test/evaluation/local.py 

4. ``` cd ./tracking ``` and run ``` python run_tracker.py atom default --dataset eotb --sequence val --epochname ATOMnet_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite ATOM if you find the work useful:
```
@inproceedings{danelljan2019atom,
  title={Atom: Accurate tracking by overlap maximization},
  author={Danelljan, Martin and Bhat, Goutam and Khan, Fahad Shahbaz and Felsberg, Michael},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4660--4669},
  year={2019}
}
```
****

##  Extended [DiMP](https://github.com/visionml/pytracking) with Our Modules
1. ``` cd dimp-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xooUnG-qrOG2v9q33m) and put it into $YOUR_PATH.

3. Change model path and other settings from line 9 to line 20 in ./lib/test/evaluation/local.py 

4. ``` cd ./tracking ``` and run ``` python run_tracker.py dimp dimp50 --dataset eotb --sequence val --epochname DIMPnet_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite DiMP if you find the work useful:
```
@inproceedings{bhat2019learning,
  title={Learning discriminative model prediction for tracking},
  author={Bhat, Goutam and Danelljan, Martin and Gool, Luc Van and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6182--6191},
  year={2019}
}
```
****
##  Extended [SparseTT](https://github.com/fzh0917/SparseTT) with Our Modules
1. ``` cd SparseTT-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xooUnG-qrOG2v9q33m) and put it into $YOUR_PATH.

3. Change model path and other settings from line 9 to line 20 in ./lib/test/evaluation/local.py 

4. ``` cd ./tracking ``` and run ``` python run_tracker.py dimp dimp50 --dataset eotb --sequence val --epochname DIMPnet_extended.pth.tar``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite DiMP if you find the work useful:
```
@inproceedings{bhat2019learning,
  title={Learning discriminative model prediction for tracking},
  author={Bhat, Goutam and Danelljan, Martin and Gool, Luc Van and Timofte, Radu},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={6182--6191},
  year={2019}
}
```
****

##  Acknowledgement
We would like to thank [PyTracking](https://github.com/visionml/pytracking),  [TransT](https://github.com/chenxin-dlut/TransT) and [Stark](https://github.com/researchmm/Stark) for providing great frameworks and toolkits.

