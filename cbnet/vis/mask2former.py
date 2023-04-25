# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Union

from torch import Tensor

from mmtrack.models.mot import BaseMultiObjectTracker
from mmtrack.registry import MODELS
from mmtrack.utils import OptConfigType, OptMultiConfig, SampleList
from mmtrack.models.vis.mask2former import Mask2Former

def add_prefix(inputs, prefix):
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value
    return outputs

@MODELS.register_module()
class CB_Mask2FormerVIS(Mask2Former):
    r"""Composite Backbone version Implementation of 
     `Masked-attention Mask
    Transformer for Universal Image Segmentation
    <https://arxiv.org/pdf/2112.01527>`_.
    """

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """Overload in order to load mmdet pretrained ckpt."""
        for key in list(state_dict):
            if key.startswith('panoptic_head'):
                state_dict[key.replace('panoptic',
                                       'track')] = state_dict.pop(key)
                
            # convert single backbone checkpoint for multiple (composite) backbone checkpoints
            if key.startswith('backbone.') and not (key.startswith('backbone.backbones.') or key.startswith('backbone.trans.')):
                for iter_id in range(self.backbone.iters):
                    state_dict[key.replace('backbone.',
                                        f'backbones.{iter_id}.')] = state_dict[key]
                del state_dict[key]

        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def loss(self, inputs: Dict[str, Tensor], data_samples: SampleList,
             **kwargs) -> Union[dict, tuple]:
        """
        Same as Mask2Former except the extra losses for all predictions from all backbones
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # shape (N * T, C, H, W)
        img = img.flatten(0, 1)

        xs = self.backbone(img)

        losses = dict()
        for i, x in enumerate(xs):
            _losses = self.track_head.loss(x, data_samples)
            losses.update(add_prefix(_losses, f'cb{i}'))
        return losses

    def predict(self,
                inputs: dict,
                data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """
        Same as Mask2Former except the extra process of backbone predictions
        """
        img = inputs['img']
        assert img.dim() == 5, 'The img must be 5D Tensor (N, T, C, H, W).'
        # the "T" is 1
        img = img.squeeze(1)
        feats = self.backbone(img)[-1] # only the predicions of lastest backbone are needed
        pred_track_ins_list = self.track_head.predict(feats, data_samples,
                                                      rescale)

        results = []
        for idx, pred_track_ins in enumerate(pred_track_ins_list):
            track_data_sample = data_samples[idx]
            track_data_sample.pred_track_instances = pred_track_ins
            results.append(track_data_sample)

        return results
