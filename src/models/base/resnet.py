import pdb
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNetFPN', 'resnet18_fpn', 'resnet34_fpn', 'resnet50_fpn', 'resnet101_fpn',
           'resnet152_fpn', 'resnext50_32x4d_fpn', 'resnext101_32x8d_fpn',
           'wide_resnet50_2_fpn', 'wide_resnet101_2_fpn']

basic_config = {
    'transer': [[64, 128, 256, 512, 256], [64, 128, 256, 256, 256]], # [x4, x8, x16, x32, x32]
    'dense': [[[0,1,2], [1,2,3], [3,4]], [0, 2, 3], [64, 128, 256]],
}

bottleneck_config = {
    'transer': [[256, 512, 1024, 2048, 256], [64, 128, 256, 256, 256]], # [x4, x8, x16, x32, x32]
    #'dense': [[[1,2,3,4], [2,3,4], [3,4]], [1, 2, 3], [128, 128, 256]],
    'dense': [[[0,1,2], [1,2,3], [3,4]], [0, 2, 3], [64, 128, 256]], 
    # 'dense': [[[1,2,3,4], [2,3,4], [3,4]], [1, 2, 3], [64, 128, 256]], # 1/4 1/2 1 zoom in model
    # 'dense': [[[0,1,2], [2,3,4]], [1, 3], [128, 256]],
}

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-f37072fd.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth',
    # 'resnet50': 'https://download.pytorch.org/models/resnet50-0676ba61.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-63fe2227.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-394f9c45.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}

def gn(planes, channel_per_group=4, max_groups=32):
    groups = planes // channel_per_group
    return nn.GroupNorm(min(groups, max_groups), planes)

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNetFPN V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        # num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_maxpool: bool = True
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.use_maxpool = use_maxpool

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        if self.use_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        # Bottom-up
        c1 = self.relu(self.bn1(self.conv1(x)))
        if self.use_maxpool:
            c1 = self.maxpool(c1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)

        return [c2, c3, c4, c5]

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)

class FPN(nn.Module):
    def __init__(
        self,
        inplanes: List[int] = [2048, 1024, 512, 256],
        outplanes: int = 256, 
        layer_idx: List[int] = [0, 3]
    ) -> None:
        super(FPN, self).__init__()

        self.layer_idx = layer_idx
        assert max(layer_idx) < len(inplanes) and min(layer_idx) >= 0, "layer_idx out of range."

        self.lateral_layers, self.smooth_layers = nn.ModuleList(), nn.ModuleList()
        for _inplanes in inplanes:
            self.lateral_layers.append(nn.Sequential(
                nn.Conv2d(_inplanes, outplanes, kernel_size=1, stride=1, padding=0, bias=False),  # Reduce channels
                nn.GroupNorm(32, outplanes),
            ))
            self.smooth_layers.append(nn.Sequential(
                nn.Conv2d(outplanes, outplanes, kernel_size=3, stride=1, padding=1, bias=False),
                nn.GroupNorm(32, outplanes),
            ))
        
        # for _idx in self.layer_idx:
        #     self.smooth_layers.append(nn.Sequential(
        #         nn.Conv2d(outplanes, outplanes, 3, 1, 1, bias=False),
        #         nn.BatchNorm2d(outplanes),
        #         nn.LeakyReLU(),
        #         nn.Conv2d(outplanes, outplanes, 3, 1, 1, bias=False),
        #     ))

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[-2:], mode='bilinear', align_corners=True) + y

    def _forward_impl(self, x: List[Tensor]) -> List[Tensor]:
        features = x[-1:0:-1]

        previous = self.smooth_layers[0](self.lateral_layers[0](features[0])) # 32x
        out = [previous] # --> [4x, 8x, 16x, 32x]
        for _i in range(1, len(features)):
            previous = self.smooth_layers[_i](F.relu(self._upsample_add(previous, self.lateral_layers[_i](features[_i]))))
            out.insert(0, previous) # fine to coarse

        return [out[i] for i in self.layer_idx]
        
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        return self._forward_impl(x)

class PPM(nn.Module):
    def __init__(
        self,
        inplane: int = 2048,
        outplane: int = 256, 
        scales: List[int] = [1, 2, 4, 8]
    ) -> None:
        super(PPM, self).__init__()

        self.trans = nn.Sequential(
            nn.Conv2d(inplane, outplane, 1, 1, bias=False),
            gn(outplane),
        )

        self.ppms = nn.ModuleList()
        for ii in scales:
            self.ppms.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(ii), 
                nn.Conv2d(outplane, outplane, 1, 1, bias=False), 
                gn(outplane),
                ))

        self.fuse = nn.Sequential(
            nn.Conv2d(outplane, outplane, 3, 1, 1, bias=False),
            gn(outplane),
        )
        self.relu = nn.ReLU(inplace=True)     

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.trans(x)
        x_size = x.size()[2:]

        out = x
        for ppm in self.ppms:
            out = torch.add(out, F.interpolate(ppm(x), x_size, mode='bilinear', align_corners=True))
        out = self.fuse(self.relu(out))

        return out
        
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class Transer(nn.Module):
    def __init__(
        self,
        inplanes: List[int] = [256, 512, 1024, 2048, 256],
        outplanes: List[int] = [64, 128, 256, 256, 256]
    ) -> None:
        super(Transer, self).__init__()

        self.transers = nn.ModuleList()
        for _inp, _oup in zip(inplanes, outplanes):
            self.transers.append(nn.Sequential(
                nn.Conv2d(_inp, _oup, 1, 1, bias=False), 
                gn(_oup), 
                ))

    def _forward_impl(self, x: List[Tensor]) -> List[Tensor]:
        out = []
        for _x, _transer in zip(x, self.transers):
            out.append(_transer(_x))
        return out
        
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        return self._forward_impl(x)

class DenseFusion(nn.Module):
    def __init__(
        self,
        inplanes: List[int] = [64, 128, 256, 256, 256], # # [0, 1, 2, 3, 4] -> [x4, x8, x16, x32, x32]
        inlevels: List[List[int]] = [[1,2,3,4], [2,3,4], [3,4]],
        outlevels: List[int] = [1, 2, 3], # [0, 1, 2, 3] -> [x4, x8, x16, x32]
        outplanes: List[int] = [128, 256, 256], 
    ) -> None:
        super(DenseFusion, self).__init__()

        self.inlevels = inlevels
        self.outlevels = outlevels

        self.trans = nn.ModuleList()
        self.merge = nn.ModuleList()
        for inlevel, outplane in zip(inlevels, outplanes):
            in_trans = nn.ModuleList()
            for _inl in inlevel:
                in_trans.append(nn.Sequential(
                    nn.Conv2d(inplanes[_inl], outplane, 1, 1, bias=False), 
                    gn(outplane)
                ))
            self.trans.append(in_trans)
            self.merge.append(nn.Sequential(
                    nn.Conv2d(outplane, outplane, 3, 1, 1, bias=False), 
                    gn(outplane)
                ))
        self.relu =nn.ReLU(inplace=True)

    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[-2:], mode='bilinear', align_corners=True) + y

    def _forward_impl(self, x: List[Tensor]) -> List[Tensor]:
        x_sizes = [_x.size()[2:] for _x in x]

        out = []
        for i, (inlevel, outlevel) in enumerate(zip(self.inlevels, self.outlevels)): # int, list, int
            in_out = 0.0
            for ii, _inl in enumerate(inlevel):
                in_out += F.interpolate(self.trans[i][ii](x[_inl]), x_sizes[outlevel], mode='bilinear', align_corners=True)
            out.append(self.merge[i](in_out))

        return out
        
    def forward(self, x: List[Tensor]) -> List[Tensor]:
        return self._forward_impl(x)

class ResNetFPN(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        # outplanes: List[int] = [128, 256, 256],  
        # layer_idx: List[int] = [0, 1, 2], 
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_maxpool: bool = True
    ) -> None:
        super(ResNetFPN, self).__init__()

        # self.outplanes = outplanes
        self.resnet = ResNet(block, layers, zero_init_residual, groups, 
                        width_per_group, replace_stride_with_dilation, norm_layer, use_maxpool)
        if block is Bottleneck:
            self.config = bottleneck_config
        else: # block is BasicBlock:
            self.config = basic_config

        self.ppm = PPM(self.config['transer'][0][-2], self.config['transer'][0][-1])
        self.transer = Transer(self.config['transer'][0], self.config['transer'][1])
        self.densefusion = DenseFusion(self.config['transer'][1], *self.config['dense'])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        x = self.resnet(x)
        x.append(self.ppm(x[-1]))
        x = self.transer(x)
        x = self.densefusion(x)
        return x

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)

def _resnetfpn(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNetFPN:
    model = ResNetFPN(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.resnet.load_state_dict(state_dict, strict=False)
    return model


def resnet18_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    return _resnetfpn('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    return _resnetfpn('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    return _resnetfpn('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    return _resnetfpn('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    return _resnetfpn('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnetfpn('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnetfpn('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    kwargs['width_per_group'] = 64 * 2
    return _resnetfpn('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2_fpn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> ResNetFPN:
    kwargs['width_per_group'] = 64 * 2
    return _resnetfpn('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
                   