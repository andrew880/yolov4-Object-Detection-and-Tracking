# YOLOv4 + Deep_SORT

<img src="https://github.com/yehengchen/Object-Detection-and-Tracking/blob/master/OneStage/yolo/deep_sort_yolov4/output/comparison.png" width="81%" height="81%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/output.gif" width="40%" height="40%"> <img src="https://github.com/yehengchen/video_demo/blob/master/video_demo/TownCentreXVID_output.gif" width="40%" height="40%">

__Object Tracking & Counting Demo - [[Social_Distance]](https://www.bilibili.com/video/BV17z4y1X7cm)  [[Pedetrain_Tracking]](https://www.bilibili.com/video/BV1Ug4y1i71w#reply3014975828)  [[Chinese Version]](https://blog.csdn.net/weixin_38107271/article/details/96741706)__
## Requirement
__Development Environment: [Deep-Learning-Environment-Setup](https://github.com/yehengchen/Ubuntu-16.04-Deep-Learning-Environment-Setup)__ 

* OpenCV
* sklean
* pillow
* numpy 1.17.0
* torch 1.7.0
* tensorflow-gpu 1.13.1
* CUDA 10.0
***

It uses:

* __Detection__: [YOLOv4](https://github.com/yehengchen/Object-Detection-and-Tracking/tree/master/OneStage/yolo/Train-a-YOLOv4-model) to detect objects on each of the video frames. - 用自己的数据训练YOLOv4模型

* __Tracking__: [Deep_SORT](https://github.com/nwojke/deep_sort) to track those objects over different frames.

*This repository contains code for Simple Online and Realtime Tracking with a Deep Association Metric (Deep SORT). We extend the original SORT algorithm to integrate appearance information based on a deep appearance descriptor. See the [arXiv preprint](https://arxiv.org/abs/1703.07402) for more information.*

## Quick Start

__0.Requirements__
*require python version: 3.6.* (recommended 3.6.10)*
    pip install -r requirements.txt
*install all the libraries currently used to run this*
    pip install -r recommends.txt
__1. Download the code to your computer.__
    
    git clone https://github.com/andrew880/yolov4-Object-Detection-and-Tracking.git
    
__2. Download [[yolov4.weights]](https://drive.google.com/drive/folders/1ynXqTi1sobqrv6gKMUCAHVIn7f1iqt9D?usp=sharing) - `mnv6`__ and place it in `model_data/`

__3. Convert the Darknet YOLO model to a Keras model:__
```
$ python convert.py model_data/yolov4.cfg model_data/yolov4.weights model_data/yolo.h5
``` 
__4. Run the YOLO_DEEP_SORT:__

```
$ python main.py -c [CLASS NAME] -i [INPUT VIDEO PATH]

$ python main.py -c person -i ./test_video/testvideo.avi

$ python camera.py
```

__5. Can change [deep_sort_yolov3/yolo.py] `__Line 100__` to your tracking object__

*DeepSORT pre-trained weights using people-ReID datasets only for person*
```
    if predicted_class != args["class"]:
               continue
    
    if predicted_class != 'person' and predicted_class != 'car':
               continue
```
## Help

* using anaconda virtual environment
```
$ conda create -n yourenvname python=x.x anaconda
$ conda activate yourenvname
$ #choose the virtual environment to run it in
$ # continue to step 0 in quick start
```
## Train on Market1501 & MARS
*People Re-identification model*

[cosine_metric_learning](https://github.com/nwojke/cosine_metric_learning) for training a metric feature representation to be used with the deep_sort tracker.

## Citation

### Deep Sort YOLOv4 :

    @misc{Object Detection and Tracking,
    title={Object Detection and Tracking},
    author={Bobby Chen},
    year={2020},
    eprint={2020},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

### YOLOv4 :

    @misc{bochkovskiy2020yolov4,
    title={YOLOv4: Optimal Speed and Accuracy of Object Detection},
    author={Alexey Bochkovskiy and Chien-Yao Wang and Hong-Yuan Mark Liao},
    year={2020},
    eprint={2004.10934},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }

### Deep_SORT :

    @inproceedings{Wojke2017simple,
    title={Simple Online and Realtime Tracking with a Deep Association Metric},
    author={Wojke, Nicolai and Bewley, Alex and Paulus, Dietrich},
    booktitle={2017 IEEE International Conference on Image Processing (ICIP)},
    year={2017},
    pages={3645--3649},
    organization={IEEE},
    doi={10.1109/ICIP.2017.8296962}
    }

    @inproceedings{Wojke2018deep,
    title={Deep Cosine Metric Learning for Person Re-identification},
    author={Wojke, Nicolai and Bewley, Alex},
    booktitle={2018 IEEE Winter Conference on Applications of Computer Vision (WACV)},
    year={2018},
    pages={748--756},
    organization={IEEE},
    doi={10.1109/WACV.2018.00087}
    }
    
## Reference
#### Github:deep_sort@[Nicolai Wojke nwojke](https://github.com/nwojke/deep_sort)
#### Github:deep_sort_yolov3@[Qidian213 ](https://github.com/Qidian213/deep_sort_yolov3)
#### Github:Deep-SORT-YOLOv4@[LeonLok](https://github.com/LeonLok/Deep-SORT-YOLOv4)