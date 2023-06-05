# CBNet meets Mask2former
An implementation of [CBNet](https://arxiv.org/pdf/2107.00420) with [Mask2former](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Masked-Attention_Mask_Transformer_for_Universal_Image_Segmentation_CVPR_2022_paper.pdf).
This project is based on [mmdetection](https://github.com/open-mmlab/mmdetection) and [mmtracking](https://github.com/open-mmlab/mmtracking). Due to version compatibility reasons, I made some small modifications to the official version of mmtracking (https://github.com/chuxiaojie/mmtrack.git) to adapt to the latest version of mmdetection.

## Installation
Step 0. Install MMEngine and MMCV using MIM.
```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```
Step 1. Install MMDetection.
```shell
mim install mmdet
pip install git+https://github.com/cocodataset/panopticapi.git
```
Step 2. Install MMTracking.
```shell
git clone https://github.com/chuxiaojie/mmtrack.git
cd mmtrack
pip install -r requirements/build.txt
pip install -v -e .  # or "python setup.py develop"
pip install git+https://github.com/JonathonLuiten/TrackEval.git
cd ..
```

## Dataset
Step 0. Please download the datasets from the official websites ([YoutubeVIS](https://youtube-vos.org/dataset/vis/), [MS-COCO](https://cocodataset.org/#download), [COCO-panoptic](http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip)). It is recommended to symlink the root of the datasets to `$mask2former/data`.
```
# For example
ln -s /data/public_data/COCO/ ./data/coco
ln -s /data/datasets/youtube_vis_2021 ./data/youtube_vis_2021
```


Step 1. Prepare youtube-vis-2021 dataset
```python
python ../mmtrack/tools/dataset_converters/youtubevis/youtubevis2coco.py -i ./data/youtube_vis_2021 -o ./data/youtube_vis_2021/annotations --version 2021
```

Check: The directory should be like this.
```none
mask2former
├── cbnet
├── configs
├── data
│   ├── coco
│   │   ├── annotations
│   │   │   ├── panoptic_train2017.json
│   │   │   ├── panoptic_train2017
│   │   │   ├── panoptic_val2017.json
│   │   │   ├── panoptic_val2017
│   │   ├── train2017
│   │   ├── val2017
│   │   ├── test2017
│   ├── youtube_vis_2021
│   │   ├── annotations
│   │   │   ├── youtube_vis_2021_test.json
│   │   │   ├── youtube_vis_2021_train.json
│   │   │   ├── youtube_vis_2021_valid.json
│   │   │── train
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │── valid
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
│   │   │── test
│   │   │   │── JPEGImages
│   │   │   │── instances.json (the official annotation files)
├── object_detection
├── video_tracking
```

## Learn to train and test
### Training
```shell
# For object detection
bash object_detection/tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
# For video tracking
bash video_tracking/tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
```
Example
```
# Train a mask2former with cb-swin-l for image panoptic segmentation on 8 GPUs.
bash object_detection/tools/dist_train.sh configs/mask2former/mask2former_cb-swin-l-p4-w12-384-in21k_8xb2-lsj-100e_coco-panoptic.py 8
# Train a mask2former with cb-swin-l for video instance segmentation on 8 GPUs.
bash video_tracking/tools/dist_train.sh configs/mask2former/mask2former_cb2-swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021.py 8
```

### Testing
```shell
# For object detection
bash object_detection/tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT_FILE} ${GPU_NUM} [optional arguments]
# For video tracking
bash video_tracking/tools/dist_test.sh ${CONFIG_FILE} ${GPU_NUM} --checkpoint ${CHECKPOINT_FILE} [optional arguments]
```
Example
```shell
# Test the mask2former with resnet-50 for image panoptic segmentation on 8 GPUs
 bash object_detection/tools/dist_test.sh object_detection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py ~/.cache/torch/hub/checkpoints/mask2former_r50_8xb2-lsj-50e_coco-panoptic_20230118_125535-54df384a.pth  8
# Test the mask2former with cb-swin-L for video instance segmentation on 1 GPU using FP16
bash video_tracking/tools/dist_test.sh configs/mask2former/$1.py 1 --checkpoint work_dirs/mask2former_cb-swin-l-p4-w12-384-in21k_8xb2-lsj-100e_coco-panoptic/iter_8000.pth --cfg-options model.track_head.pixel_decoder.encoder.layer_cfg.self_attn_cfg.im2col_step=1 model.init_cfg=None model.backbone.init_cfg=None test_cfg.fp16=true
```

### Inference VIS models
This script can inference an input video / images with a video instance segmentation model.

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python video_tracking/demo/demo_vis.py \
    ${CONFIG_FILE} \
    --input ${INPUT} \
    [--output ${OUTPUT}] \
    [--checkpoint ${CHECKPOINT_FILE}] \
    [--score-thr ${SCORE_THR} \
    [--device ${DEVICE}] \
    [--backend ${BACKEND}] \
    [--show]
```

The `INPUT` and `OUTPUT` support both mp4 video format and the folder format.

Optional arguments:

- `OUTPUT`: Output of the visualized demo. If not specified, the `--show` is obligate to show the video on the fly.
- `CHECKPOINT_FILE`: The checkpoint is optional in case that you already set up the pretrained models in the config by the key `pretrains`.
- `SCORE_THR`: The threshold of score to filter bboxes.
- `DEVICE`: The device for inference. Options are `cpu` or `cuda:0`, etc.
- `BACKEND`: The backend to visualize the boxes. Options are `cv2` and `plt`.
- `--show`: Whether show the video on the fly.


Examples of running vis model:

Assume that you have already downloaded the checkpoints to the directory `work_dirs/`

```shell
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python video_tracking/demo/demo_vis.py \
    configs/mask2former/mask2former_cb2-swin-b-p4-w12-384-in21k_8xb2-8e_youtubevis2021.py \
    --input ${VIDEO_FILE} \
    --checkpoint work_dirs/mask2former_cb2-swin-b-p4-w12-384-in21k_8xb2-8e_youtubevis2021/iter_8000.pth \
    --output ${OUTPUT} \
```

## Citing CBNet
If you use our code/model, please consider to cite our paper [CBNet: A Composite Backbone Network Architecture for Object Detection](https://arxiv.org/pdf/2107.00420).
```
@ARTICLE{9932281,
  author={Liang, Tingting and Chu, Xiaojie and Liu, Yudong and Wang, Yongtao and Tang, Zhi and Chu, Wei and Chen, Jingdong and Ling, Haibin},
  journal={IEEE Transactions on Image Processing}, 
  title={CBNet: A Composite Backbone Network Architecture for Object Detection}, 
  year={2022},
  volume={31},
  pages={6893-6906},
  doi={10.1109/TIP.2022.3216771}}
```

