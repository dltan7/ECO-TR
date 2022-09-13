"""
Backbone modules.
"""
import torch
import torch.nn.functional as F
from torch import nn
from typing import List

from .misc import NestedTensor

from .position_encoding import build_position_encoding

from ..base import resnet as models
# from .base import resnet_fpn as models

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.body = backbone 


    def forward(self, tensor_list: NestedTensor):
        h,w = tensor_list.tensors.shape[2:4]
        left = self.body(tensor_list.tensors[..., :w//2])
        right = self.body(tensor_list.tensors[..., w//2:])
        
        left_and_right = []
        for _l, _r in zip(left, right):
            left_and_right.append(torch.cat([_l, _r], dim=-1))

        assert tensor_list.mask is not None
        out = [NestedTensor(_lr, \
                    F.interpolate(tensor_list.mask[None].float(), \
                    size=_lr.shape[-2:]).to(torch.bool)[0]) \
                    for _lr in left_and_right]

        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 dilation: bool,
                 use_maxpool: bool = True):
        backbone = getattr(models, name)()
        super().__init__(backbone)


class Joiner(nn.Module):
    def __init__(self, backbone, pos_embeds):
        super().__init__()
        self.backbone = backbone
        self.pos_embeds = pos_embeds

    def forward(self, tensor_list: NestedTensor):
        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for i, x in enumerate(xs):
            out.append(x)
            # position encoding
            pos.append(self.pos_embeds[i](x).to(x.tensors.dtype))
        return out, pos

def build_backbone(args):
    backbone = Backbone(args.backbone, args.dilation, use_maxpool=True) 
    pos_embeds = nn.ModuleList()
    for _dim in backbone.body.config['dense'][-1]:
        pos_embeds.append(build_position_encoding(_dim, args.position_embedding))
    model = Joiner(backbone, pos_embeds)
    return model