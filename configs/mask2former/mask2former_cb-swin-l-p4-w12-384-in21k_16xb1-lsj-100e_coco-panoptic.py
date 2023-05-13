_base_ = [
    '../../object_detection/configs/mask2former/mask2former_swin-l-p4-w12-384-in21k_16xb1-lsj-100e_coco-panoptic.py'
]

custom_imports = dict(imports=['cbnet'], allow_failed_imports=False)
depths = [2, 2, 18, 2]
cb_iters = 2
model = dict(type='CB_Mask2Former',
             backbone=dict(
                 type='CB_SwinTransformer',
                 iters=cb_iters,
             ))

# set all layers in backbone to lr_mult=0.1
# set all norm layers, position_embeding,
# query_embeding, level_embeding to decay_multi=0.0
backbone_norm_multi = dict(lr_mult=0.1, decay_mult=0.0)
backbone_embed_multi = dict(lr_mult=0.1, decay_mult=0.0)
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
custom_keys = {
    'backbone': dict(lr_mult=0.1, decay_mult=1.0),
    'backbone.patch_embed.norm': backbone_norm_multi,
    'backbone.norm': backbone_norm_multi,
    'absolute_pos_embed': backbone_embed_multi,
    'relative_position_bias_table': backbone_embed_multi,
    'query_embed': embed_multi,
    'query_feat': embed_multi,
    'level_embed': embed_multi
}
custom_keys.update({
    f'backbone.backbones.{iter_id}': dict(lr_mult=0.1, decay_mult=1.0)
    for iter_id in range(cb_iters)
})
custom_keys.update({
    f'backbone.backbones.{iter_id}.{module}': backbone_norm_multi
    for module in ['patch_embed.norm', 'norm'] for iter_id in range(cb_iters)
})
custom_keys.update({
    f'backbone.backbones.{iter_id}.{module}': backbone_embed_multi
    for module in ['absolute_pos_embed', 'relative_position_bias_table']
    for iter_id in range(cb_iters)
})
custom_keys.update({
    f'backbone.backbones.{iter_id}.stages.{stage_id}.blocks.{block_id}.norm':
    backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks) for iter_id in range(cb_iters)
})
custom_keys.update({
    f'backbone.backbones.{iter_id}.stages.{stage_id}.downsample.norm':
    backbone_norm_multi
    for stage_id in range(len(depths) - 1) for iter_id in range(cb_iters)
})

# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
