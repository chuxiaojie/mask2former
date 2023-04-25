_base_ = './mask2former_r50_8xb2-8e_youtubevis2021.py'

checkpoint='https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_r50_8xb2-lsj-50e_coco/mask2former_r50_8xb2-lsj-50e_coco_20220506_191028-41b088b6.pth'
model = dict(init_cfg=dict(checkpoint=checkpoint))