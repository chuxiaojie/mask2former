_base_ = './mask2former_swin-s-p4-w7-224_8xb2-8e_youtubevis2021.py'

checkpoint='https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-s-p4-w7-224_8xb2-lsj-50e_coco_20220504_001756-c9d0c4f2.pth'
model = dict(init_cfg=dict(checkpoint=checkpoint))