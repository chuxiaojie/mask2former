import torch
from torch import nn
from mmengine.model import constant_init
from torch.nn import functional as F


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DHLC(nn.Module):
    def __init__(self, in_channels, out_channels, zero_init=True):
        super().__init__()
        assert len(in_channels) <= len(out_channels)
        m = len(out_channels)
        n = len(in_channels)
        self.linears = nn.ModuleList()
        self.zero_init = zero_init
        self.out_channels=out_channels
        self.split_sizes=[out_channels[:i+1+m-n] for i in range(n)]
        for i,in_c in enumerate(in_channels):
            out_c = sum(self.split_sizes[i])
            linear = nn.Conv2d(in_c, out_c, 1) if in_c and out_c else None
            self.linears.append(linear)
        self.init_weights()

    def init_weights(self):
        if self.zero_init:
            for m in self.linears:
                constant_init(m, 0)

    def forward(self, feats):
        assert len(feats) == len(self.linears) <= len(self.out_channels)
        outs = [m(x).split(sz,dim=1) if m else [[] for _ in sz] for m,x,sz in zip(self.linears, feats, self.split_sizes)]
        cb_feats = []
        for i in range(len(self.out_channels)):
            feeds = [ ys[i] for ys in outs[i:] if ys[i] != [] and ys[i].shape[1]] 
            cb_feats.append(feeds)
        return cb_feats

CONN_LAYERS = {
    'DHLC': DHLC,
}

def build_conn_layer(cfg, in_channels, out_channels):
    if not isinstance(cfg, dict):
        raise TypeError('cfg must be a dict')
    if 'type' not in cfg:
        raise KeyError('the cfg dict must contain the key "type"')
    cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONN_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')

    conn_layer = CONN_LAYERS.get(layer_type)

    return conn_layer(in_channels=in_channels, out_channels=out_channels, **cfg_)