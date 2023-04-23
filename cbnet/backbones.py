# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

from mmdet.registry import MODELS
from mmdet.models.backbones.resnet import ResNet as _ResNet
from mmdet.models.backbones.resnext import ResNeXt as _ResNeXt
from mmdet.models.backbones.res2net import Res2Net as _Res2Net
from mmdet.models.backbones.swin import SwinTransformer as _SwinTransformer
from .connection import build_conn_layer

class ResNetBase:
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            if self.deep_stem and hasattr(self, 'stem'):
                self.stem.eval()
                for param in self.stem.parameters():
                    param.requires_grad = False
            elif hasattr(self, 'conv1'):
                self.norm1.eval()
                for m in [self.conv1, self.norm1]:
                    for param in m.parameters():
                        param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            if not hasattr(self, f'layer{i}'):
                continue
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def del_stages(self, index):
        if index >= 0:
            if self.deep_stem:
                del self.stem
            else:
                del self.conv1
                delattr(self, self.norm1_name)
            del self.maxpool
        
        for i in range(1, index+1):
            delattr(self, f'layer{i}')

    def forward(self, x, extra_feats=None):
        """Forward function."""
        outs = []
        rec = extra_feats is not None

        if not rec:
            """Forward function."""
            if self.deep_stem:
                x = self.stem(x)
            else:
                x = self.conv1(x)
                x = self.norm1(x)
                x = self.relu(x)
            x = self.maxpool(x)

        for i, layer_name in enumerate(self.res_layers):
            if rec:
                if i < self.fuse_stage:
                    continue
                else:
                    cb_fs = [F.interpolate(f, size=x.shape[2:], mode='nearest') for f in extra_feats[i-self.fuse_stage]]
                    x = torch.sum(torch.stack([x]+cb_fs), dim=0)
            elif i==self.fuse_stage: # not rec
                new_x = x
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        if rec:
            return tuple(outs)
        else:
            return tuple(outs), new_x

class ResNet(ResNetBase, _ResNet):
    pass
    
class Res2Net(ResNetBase, _Res2Net):
    pass
    
class ResNeXt(ResNetBase, _ResNeXt):
    pass
    

class CBase(BaseModule):
    def __init__(self, net, conn_cfg=dict(type='DHLC'), iters=2, share_weights=False, fuse_stage=1, last_out_only=False, **kwargs):
        super().__init__()
        self.iters = iters
        self.last_out_only = last_out_only # last backbone's output only
        self.trans = nn.ModuleList()
        self.backbones = nn.ModuleList()
        for i in range(iters):
            if i==0 or not share_weights:
                backbone = net(**kwargs) 
            backbone.fuse_stage = fuse_stage
            if i>0:
                in_channels, out_channels = self.get_trans_channels(backbone)
                if not share_weights and hasattr(backbone, 'del_stages'):
                    backbone.del_stages(fuse_stage)
                trans = build_conn_layer(conn_cfg, in_channels, out_channels)
                self.trans.append(trans)
                
            self.backbones.append(backbone)
    
    def get_trans_channels(self, backbone):
        raise NotImplementedError

    def init_weights(self, *args, **kwargs):
        for m in self.backbones:
            m.init_weights(*args, **kwargs)

    def forward(self, x):
        outs, x = self.backbones[0](x)
        outs_list = [outs]
        for (B,T) in zip(self.backbones[1:], self.trans):
            outs = B(x, T(outs[max(B.fuse_stage,0):]))
            outs = list(outs_list[-1][:B.fuse_stage])+list(outs)
            outs_list.append(tuple(outs))
        if self.last_out_only:
            return outs_list[-1]
        else:
            return outs_list


@MODELS.register_module()
class CB_ResNet(CBase):
    def __init__(self, *args, **kwargs):
        super().__init__(net=ResNet, *args, **kwargs)

    def get_trans_channels(self, backbone):
        out_channels = [backbone.stem_channels] + [getattr(backbone, f'layer{i + 1}')[0].downsample[0].in_channels for i in range(backbone.num_stages)]
        in_channels = [getattr(backbone, f'layer{i + 1}')[0].downsample[0].out_channels for i in range(backbone.num_stages)]
        return in_channels[max(backbone.fuse_stage,0):], out_channels[backbone.fuse_stage+1:]

@MODELS.register_module()
class CB_ResNeXt(CBase):
    def __init__(self, *args, **kwargs):
        super().__init__(net=ResNeXt, *args, **kwargs)

    def get_trans_channels(self, backbone):
        out_channels = [backbone.stem_channels] + [getattr(backbone, f'layer{i + 1}')[0].downsample[0].in_channels for i in range(backbone.num_stages)]
        in_channels = [getattr(backbone, f'layer{i + 1}')[0].downsample[0].out_channels for i in range(backbone.num_stages)]
        return in_channels[max(backbone.fuse_stage,0):], out_channels[backbone.fuse_stage+1:]


@MODELS.register_module()
class CB_Res2Net(CBase):
    def __init__(self, *args, **kwargs):
        super().__init__(net=Res2Net, *args, **kwargs)

    def get_trans_channels(self, backbone):
        out_channels = [backbone.stem_channels] + [getattr(backbone, f'layer{i + 1}')[0].downsample[1].in_channels for i in range(backbone.num_stages)]
        in_channels = [getattr(backbone, f'layer{i + 1}')[0].downsample[1].out_channels for i in range(backbone.num_stages)]
        return in_channels[max(backbone.fuse_stage,0):], out_channels[backbone.fuse_stage+1:]


class RecursiveBase(BaseModule):
    """
    Simple CBNet without pruning
    """
    def __init__(self, net, conn_cfg=dict(type='DHLC'), iters=2, share_weights=False, **kwargs):
        super().__init__()
        self.iters = iters
        self.trans = nn.ModuleList()
        self.backbones = nn.ModuleList()
        for i in range(iters):
            if i==0 or not share_weights:
                backbone = net(**kwargs) 
                in_channels, out_channels = self.get_trans_channels(backbone)
            if i>0:
                trans = build_conn_layer(conn_cfg, in_channels, out_channels)
                self.trans.append(trans)
                
            self.backbones.append(backbone)
    
    def get_trans_channels(self, backbone):
        raise NotImplementedError

    def init_weights(self, *args, **kwargs):
        for m in self.backbones:
            m.init_weights(*args, **kwargs)

    def forward(self, x):
        outs = self.backbones[0](x)
        outs_list = [outs]
        for (B,T) in zip(self.backbones[1:], self.trans):
            outs = B(x, T(outs))
            outs_list.append(outs)
        return outs_list
    

def spatial_interpolate(x, H, W):
    B, C = x.shape[:2]
    if H != x.shape[2] or W != x.shape[3]:
        x = F.interpolate(x, size=(H, W), mode='nearest')
    x = x.view(B, C, -1).permute(0, 2, 1).contiguous()  # B, T, C
    return x

class SwinTransformer(_SwinTransformer):
    def forward(self, x, extra_feats=None):
        x, hw_shape = self.patch_embed(x)

        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.stages):
            if extra_feats is not None:
                fs = [spatial_interpolate(f, *hw_shape) for f in extra_feats[i]]
                x = torch.sum(torch.stack([x]+fs), dim=0)
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs

@MODELS.register_module()
class CB_SwinTransformer(RecursiveBase):
    def __init__(self, *args, **kwargs):
        super().__init__(net=SwinTransformer, *args, **kwargs)

    def get_trans_channels(self, backbone):
        return backbone.num_features, backbone.num_features