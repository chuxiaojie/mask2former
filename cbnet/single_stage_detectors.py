from mmdet.registry import MODELS
from mmdet.models.detectors import MaskFormer, Mask2Former
from typing import Dict, Tuple, Union
from torch import Tensor
from mmdet.structures import OptSampleList, SampleList

def add_prefix(inputs, prefix):
    outputs = dict()
    for name, value in inputs.items():
        outputs[f'{prefix}.{name}'] = value
    return outputs

class CB_SingleStageDetector:
    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        """Extract features.

        Args:
            batch_inputs (Tensor): Image tensor with shape (N, C, H ,W).

        Returns:
            tuple[Tensor]: Multi-level features that may have
            different resolutions.
        """
        xs = self.backbone(batch_inputs)
        if not self.training:
            xs = xs[-1:]

        if self.with_neck:
            xs = [self.neck(x) for x in xs]

        if self.training:
            return xs
        else:
            return xs[-1]
        
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        xs = self.extract_feat(batch_inputs)
        losses = dict()
        for i, x in enumerate(xs):
            _losses = self.bbox_head.loss(x, batch_data_samples)
            losses.update(add_prefix(_losses, f'cb{i}'))
        return losses

class CB_MaskFormer(CB_SingleStageDetector):
    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        xs = self.extract_feat(batch_inputs)
        losses = dict()
        for i, x in enumerate(xs):
            _losses = self.panoptic_head.loss(x, batch_data_samples)
            losses.update(add_prefix(_losses, f'cb{i}'))
        return losses
    
@MODELS.register_module()
class CB_MaskFormer(CB_MaskFormer, MaskFormer):
    pass

@MODELS.register_module()
class CB_Mask2Former(CB_MaskFormer, Mask2Former):
    pass