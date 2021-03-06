# Boosting Tracking Performance by Introducing Event-based Vision Sensor

<p align="center">
  <a href="https://youtu.be/712e-zdCcDg">
    <img src="video-png.png" alt="Video to Events" width="600"/>
  </a>
</p>

<!-- ##  Example Code Evaluation 
 
The code is built on [visionml/pytracking](https://github.com/visionml/pytracking)  and tested on Ubuntu 18.04 environment with RTX 3090 GPUs. -->

## Download DATASET
Download our proposed [dataset](https://zhangjiqing.com/dataset/)

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

##  Extended [PrDiMP](https://github.com/visionml/pytracking) with Our Modules
1. ``` cd prdimp-fusion ```

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
##  Extended [SparseTT](https://github.com/fzh0917/SparseTT) with Our Modules
1. ``` cd SparseTT-fusion ```

2. Download the [pretrained model](https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth)(Swin backbone) and put it into models/swin/

3. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xooXPkpqyV-OdCSA-F?e=7RTRSo) and put it into work_dir/train/sparsett-swin-got10k-train/.

4. Change model path and other settings in experiments/sparsett/test/got10k/sparsett_swin_got10k.yaml line 12 and line 68-74.

5. run ``` python main/test.py --config experiments/sparsett/test/got10k/sparsett_swin_got10k.yaml``` the predicted bounding boxes are be saved in work_dir/test/sparsett-swin-got10k-test/result/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite SparseTT if you find the work useful:
```
@article{fu2022sparsett,
  title={SparseTT: Visual Tracking with Sparse Transformers},
  author={Fu, Zhihong and Fu, Zehua and Liu, Qingjie and Cai, Wenrui and Wang, Yunhong},
  booktitle={IJCAI},
  year={2022}
}
```
****
##  Extended [TrDiMP](https://github.com/594422814/TransformerTrack) with Our Modules
1. ``` cd trdimp-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzWxILiiehXLXC6O?e=jqahvi) and put it into ./checkpoints/ltr/trdimp/trdimp.

3. Change dataset and model path from line 8 to line 10 in pytracking/evaluation/local.py 

4. ``` cd pytracking ``` and run ``` python run_tracker.py trdimp trdimp --dataset eotb --sequence val ``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite trdimp if you find the work useful:
```
@inproceedings{wang2021transformer,
  title={Transformer meets tracker: Exploiting temporal context for robust visual tracking},
  author={Wang, Ning and Zhou, Wengang and Wang, Jie and Li, Houqiang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={1571--1580},
  year={2021}
}
```
****

##  Extended [ToMP](https://github.com/visionml/pytracking) with Our Modules
1. ``` cd ToMP-fusion ```

2. Download the [pretrained model](https://1drv.ms/u/s!AoopRFuuZ7xohzWxILiiehXLXC6O?e=jqahvi) and put it into ./checkpoints/ltr/tomp/tomp.

3. Change dataset and model path in pytracking/evaluation/local.py 

4. ``` cd pytracking ``` and run ``` python run_tracker.py tomp tomp50 --dataset eotb --sequence val ``` the predicted bounding boxes are be saved in pytracking/trakcing_results/.  
    - The predicted  bounding box format:  An N×4 matrix with each line representing object location [xmin, ymin, width, height].

Please cite trdimp if you find the work useful:
```
@inproceedings{mayer2022transforming,
  title={Transforming model prediction for tracking},
  author={Mayer, Christoph and Danelljan, Martin and Bhat, Goutam and Paul, Matthieu and Paudel, Danda Pani and Yu, Fisher and Van Gool, Luc},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={8731--8740},
  year={2022}
}
```
****

##  Acknowledgement
We would like to thank [PyTracking](https://github.com/visionml/pytracking),  [PySOT](https://github.com/STVIR/pysot) and [video_analyst](https://github.com/megvii-research/video_analyst) for providing great frameworks and toolkits.

