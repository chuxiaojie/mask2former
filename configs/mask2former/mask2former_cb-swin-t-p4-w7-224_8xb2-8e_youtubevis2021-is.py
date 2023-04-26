_base_ = './mask2former_cb-swin-t-p4-w7-224_8xb2-8e_youtubevis2021.py'

checkpoint='https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco/mask2former_swin-t-p4-w7-224_8xb2-lsj-50e_coco_20220508_091649-01b0f990.pth'
model = dict(init_cfg=dict(checkpoint=checkpoint))