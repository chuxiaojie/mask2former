_base_ = ['./mask2former_swin-l-p4-w12-384-in21k_8xb2-8e_youtubevis2021.py']
depths = [2, 2, 18, 2]
model = dict(
    backbone=dict(
        embed_dims=128,
        num_heads=[4, 8, 16, 32],
        window_size=12,
    ),
    track_head=dict(
        type='Mask2FormerHead',
        in_channels=[128, 256, 512, 1024],
        num_queries=100),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=  # noqa: E251
        'https://download.openmmlab.com/mmdetection/v3.0/mask2former/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic/mask2former_swin-b-p4-w12-384-in21k_8xb2-lsj-50e_coco-panoptic_20220329_230021-05ec7315.pth'))

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
    f'backbone.stages.{stage_id}.blocks.{block_id}.norm': backbone_norm_multi
    for stage_id, num_blocks in enumerate(depths)
    for block_id in range(num_blocks)
})
custom_keys.update({
    f'backbone.stages.{stage_id}.downsample.norm': backbone_norm_multi
    for stage_id in range(len(depths) - 1)
})
# optimizer
optim_wrapper = dict(
    paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0))
