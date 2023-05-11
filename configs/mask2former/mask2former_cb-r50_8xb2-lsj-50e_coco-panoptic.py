_base_ = [
    '../../object_detection/configs/mask2former/mask2former_r50_8xb2-lsj-50e_coco-panoptic.py'
]

custom_imports = dict(imports=['cbnet'], allow_failed_imports=False)
cb_iters = 2
model = dict(type='CB_Mask2Former',
             backbone=dict(
                 type='CB_ResNet',
                 iters=cb_iters,
             ))
