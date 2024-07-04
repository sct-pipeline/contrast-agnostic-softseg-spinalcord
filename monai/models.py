import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ---------------------------- Imports for nnUNet's Model -----------------------------
from dynamic_network_architectures.architectures.unet import PlainConvUNet, ResidualEncoderUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from dynamic_network_architectures.initialization.weight_init import init_last_bn_before_add_to_0


# ======================================================================================================
#                              Define plans json taken from nnUNet
# ======================================================================================================
nnunet_plans = {
    "UNet_class_name": "PlainConvUNet",
    "UNet_base_num_features": 32,
    "n_conv_per_stage_encoder": [2, 2, 2, 2, 2, 2],
    "n_conv_per_stage_decoder": [2, 2, 2, 2, 2],
    "pool_op_kernel_sizes": [
        [1, 1, 1],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [2, 2, 2],
        [1, 2, 2]
    ],
    "conv_kernel_sizes": [
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3],
        [3, 3, 3]
    ],
    "unet_max_num_features": 320,
}


# ======================================================================================================
#                               Utils for nnUNet's Model
# ====================================================================================================
class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)


# ======================================================================================================
#                               Define the network based on plans json
# ====================================================================================================
def create_nnunet_from_plans(plans, num_input_channels: int, num_classes: int, deep_supervision: bool = True):
    """
    Adapted from nnUNet's source code: 
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9

    """
    num_stages = len(plans["conv_kernel_sizes"])

    dim = len(plans["conv_kernel_sizes"][0])
    conv_op = convert_dim_to_conv_op(dim)

    segmentation_network_class_name = plans["UNet_class_name"]
    mapping = {
        'PlainConvUNet': PlainConvUNet,
        'ResidualEncoderUNet': ResidualEncoderUNet
    }
    kwargs = {
        'PlainConvUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        },
        'ResidualEncoderUNet': {
            'conv_bias': True,
            'norm_op': get_matching_instancenorm(conv_op),
            'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
            'dropout_op': None, 'dropout_op_kwargs': None,
            'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
        }
    }
    assert segmentation_network_class_name in mapping.keys(), 'The network architecture specified by the plans file ' \
                                                              'is non-standard (maybe your own?). Yo\'ll have to dive ' \
                                                              'into either this ' \
                                                              'function (get_network_from_plans) or ' \
                                                              'the init of your nnUNetModule to accomodate that.'
    network_class = mapping[segmentation_network_class_name]

    conv_or_blocks_per_stage = {
        'n_conv_per_stage'
        if network_class != ResidualEncoderUNet else 'n_blocks_per_stage': plans["n_conv_per_stage_encoder"],
        'n_conv_per_stage_decoder': plans["n_conv_per_stage_decoder"]
    }
    
    # network class name!!
    model = network_class(
        input_channels=num_input_channels,
        n_stages=num_stages,
        features_per_stage=[min(plans["UNet_base_num_features"] * 2 ** i, 
                                plans["unet_max_num_features"]) for i in range(num_stages)],
        conv_op=conv_op,
        kernel_sizes=plans["conv_kernel_sizes"],
        strides=plans["pool_op_kernel_sizes"],
        num_classes=num_classes,    
        deep_supervision=deep_supervision,
        **conv_or_blocks_per_stage,
        **kwargs[segmentation_network_class_name]
    )
    model.apply(InitWeights_He(1e-2))
    if network_class == ResidualEncoderUNet:
        model.apply(init_last_bn_before_add_to_0)
    
    return model


# ======================================================================================================
# 3D Anisotropic Hybrid Network (AH-Net) https://github.com/lsqshr/AH-Net/
# Used in BigAug: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7393676/
# ====================================================================================================

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def conv3x3x1(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=(3, 3, 1),
        stride=stride,
        padding=(1, 1, 0),
        bias=False)


class BasicBlock3x3x1(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3x1, self).__init__()
        self.conv1 = conv3x3x1(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x1(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock3x3x3(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock3x3x3, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck3x3x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck3x3x1, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=(3, 3, 1),
            stride=stride,
            padding=(1, 1, 0),
            bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pool = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
            if out.size() != residual.size():
                out = self.pool(out)

        out += residual
        out = self.relu(out)

        return out


class Projection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Projection, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))


class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate,
                 drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = Pseudo3DLayer(num_input_features + i * growth_rate,
                                  growth_rate, bn_size, drop_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class UpTransition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(UpTransition, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2, mode='trilinear'))


class Final(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(Final, self).__init__()
        self.add_module('norm', nn.BatchNorm3d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module(
            'conv',
            nn.Conv3d(
                num_input_features,
                num_output_features,
                kernel_size=(3, 3, 1),
                stride=1,
                padding=(1, 1, 0),
                bias=False))
        self.add_module('pool', nn.Upsample(scale_factor=2, mode='trilinear'))


class Pseudo3DLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(Pseudo3DLayer, self).__init__()
        # 1x1x1
        self.bn1 = nn.BatchNorm3d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(
            num_input_features,
            bn_size * growth_rate,
            kernel_size=1,
            stride=1,
            bias=False)

        # 3x3x1
        self.bn2 = nn.BatchNorm3d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(
            bn_size * growth_rate,
            growth_rate,
            kernel_size=(3, 3, 1),
            stride=1,
            padding=(1, 1, 0),
            bias=False)

        # 1x1x3
        self.bn3 = nn.BatchNorm3d(growth_rate)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv3d(
            growth_rate,
            growth_rate,
            kernel_size=(1, 1, 3),
            stride=1,
            padding=(0, 0, 1),
            bias=False)

        # 1x1x1
        self.bn4 = nn.BatchNorm3d(growth_rate)
        self.relu4 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv3d(
            growth_rate, growth_rate, kernel_size=1, stride=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        inx = x
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv1(x)

        x = self.bn2(x)
        x = self.relu2(x)
        x3x3x1 = self.conv2(x)

        x = self.bn3(x3x3x1)
        x = self.relu3(x)
        x1x1x3 = self.conv3(x)

        x = x3x3x1 + x1x1x3
        x = self.bn4(x)
        x = self.relu4(x)
        new_features = self.conv4(x)

        self.drop_rate = 0  # Dropout will make trouble!
                            # since we use the train mode for inference
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training)
        return torch.cat([inx, new_features], 1)


class PVP(nn.Module):
    def __init__(self, in_ch):
        super(PVP, self).__init__()
        self.pool64 = nn.MaxPool3d(kernel_size=(64, 64, 1), stride=(64, 64, 1))
        self.pool32 = nn.MaxPool3d(kernel_size=(32, 32, 1), stride=(32, 32, 1))
        self.pool16 = nn.MaxPool3d(kernel_size=(16, 16, 1), stride=(16, 16, 1))
        self.pool8 = nn.MaxPool3d(kernel_size=(8, 8, 1), stride=(8, 8, 1))

        self.proj64 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj32 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj16 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))
        self.proj8 = nn.Conv3d(
            in_ch, 1, kernel_size=(1, 1, 1), stride=1, padding=(1, 1, 0))

    def forward(self, x):
        x64 = F.upsample(
            self.proj64(self.pool64(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x32 = F.upsample(
            self.proj32(self.pool32(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x16 = F.upsample(
            self.proj16(self.pool16(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x8 = F.upsample(
            self.proj8(self.pool8(x)),
            size=(x.size(2), x.size(3), x.size(4)),
            mode='trilinear')
        x = torch.cat((x64, x32, x16, x8), dim=1)
        return x


class AHNet(nn.Module):
    def __init__(self, layers=[3, 4, 6, 3], num_classes=1000):
        self.inplanes = 64
        super(AHNet, self).__init__()

        # Make the 3x3x1 resnet layers
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=(7, 7, 3),
            stride=(2, 2, 1),
            padding=(3, 3, 1),
            bias=False)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 1, 2), stride=(1, 1, 2))
        self.bn0 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        self.layer1 = self._make_layer(
            Bottleneck3x3x1, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(
            Bottleneck3x3x1, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(
            Bottleneck3x3x1, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(
            Bottleneck3x3x1, 512, layers[3], stride=2)

        # Make the 3D dense decoder layers
        DENSEGROWTH = 20
        DENSEBN = 4
        NDENSELAYER = 3

        num_init_features = 64
        NOUTRES1 = 256
        NOUTRES2 = 512
        NOUTRES3 = 1024
        NOUTRES4 = 2048

        self.up0 = UpTransition(NOUTRES4, NOUTRES3)
        self.dense0 = DenseBlock(
            num_layers=NDENSELAYER,
            num_input_features=NOUTRES3,
            bn_size=DENSEBN,
            growth_rate=DENSEGROWTH,
            drop_rate=0.0)
        NOUTDENSE0 = NOUTRES3 + NDENSELAYER * DENSEGROWTH

        self.up1 = UpTransition(NOUTDENSE0, NOUTRES2)
        self.dense1 = DenseBlock(
            num_layers=NDENSELAYER,
            num_input_features=NOUTRES2,
            bn_size=DENSEBN,
            growth_rate=DENSEGROWTH,
            drop_rate=0.0)
        NOUTDENSE1 = NOUTRES2 + NDENSELAYER * DENSEGROWTH

        self.up2 = UpTransition(NOUTDENSE1, NOUTRES1)
        self.dense2 = DenseBlock(
            num_layers=NDENSELAYER,
            num_input_features=NOUTRES1,
            bn_size=DENSEBN,
            growth_rate=DENSEGROWTH,
            drop_rate=0.0)
        NOUTDENSE2 = NOUTRES1 + NDENSELAYER * DENSEGROWTH

        self.trans1 = Projection(NOUTDENSE2, num_init_features)
        self.dense3 = DenseBlock(
            num_layers=NDENSELAYER,
            num_input_features=num_init_features,
            bn_size=DENSEBN,
            growth_rate=DENSEGROWTH,
            drop_rate=0.0)
        NOUTDENSE3 = num_init_features + DENSEGROWTH * NDENSELAYER

        self.up3 = UpTransition(NOUTDENSE3, num_init_features)
        self.dense4 = DenseBlock(
            num_layers=NDENSELAYER,
            num_input_features=num_init_features,
            bn_size=DENSEBN,
            growth_rate=DENSEGROWTH,
            drop_rate=0.0)
        NOUTDENSE4 = num_init_features + DENSEGROWTH * NDENSELAYER

        self.psp = PVP(NOUTDENSE4)
        self.final = Final(4 + NOUTDENSE4, 1)

        # Initialise parameters
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=(stride, stride, 1),
                    bias=False),
                nn.MaxPool3d(
                    kernel_size=(1, 1, stride), stride=(1, 1, stride)),
                nn.BatchNorm3d(planes * block.expansion), )

        layers = []
        layers.append(
            block(self.inplanes, planes, (stride, stride, 1), downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        sum0 = self.up0(fm4) + fm3
        d0 = self.dense0(sum0)

        sum1 = self.up1(d0) + fm2
        d1 = self.dense1(sum1)

        sum2 = self.up2(d1) + fm1
        d2 = self.dense2(sum2)

        sum3 = self.trans1(d2) + pool_x
        d3 = self.dense3(sum3)

        sum4 = self.up3(d3) + conv_x
        d4 = self.dense4(sum4)

        psp = self.psp(d4)
        x = torch.cat((psp, d4), dim=1)
        return self.final(x)

    def copy_from(self, net):
        # Copy the initial module CONV1 -- Need special care since
        # we only have one input channel in the 3D network
        p2d, p3d = next(net.conv1.parameters()), next(self.conv1.parameters())

        # From 64x3x7x7 -> 64x3x7x7x1 -> 64x1x7x7x3
        p3d.data = p2d.data.unsqueeze(dim=4).permute(0, 4, 2, 3, 1).clone()

        # Copy the initial module BN1
        copy_bn_param(net.bn0, self.bn0)

        # Copy layer1
        layer_2D = []
        layer_3D = []
        for m1 in net.layer1.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer1.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer2
        layer_2D = []
        layer_3D = []
        for m1 in net.layer2.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer2.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer3
        layer_2D = []
        layer_3D = []
        for m1 in net.layer3.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer3.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)

        # Copy layer4
        layer_2D = []
        layer_3D = []
        for m1 in net.layer4.modules():
            if isinstance(m1, nn.Conv2d) or isinstance(m1, nn.BatchNorm2d):
                layer_2D.append(m1)

        for m1 in self.layer4.modules():
            if isinstance(m1, nn.Conv3d) or isinstance(m1, nn.BatchNorm3d):
                layer_3D.append(m1)

        for m1, m2 in zip(layer_2D, layer_3D):
            if isinstance(m1, nn.Conv2d):
                copy_conv_param(m1, m2)
            if isinstance(m1, nn.BatchNorm2d):
                copy_bn_param(m1, m2)


def copy_conv_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data.unsqueeze(dim=4).clone()[:]


def copy_bn_param(module2d, module3d):
    for p2d, p3d in zip(module2d.parameters(), module3d.parameters()):
        p3d.data[:] = p2d.data[:]  # Two parameter gamma and beta


# ======================================================================================================
# 2D AH-Net Model
# ====================================================================================================


class GCN(nn.Module):
    '''
    The Global Convolutional Network module using large 1D
    Kx1 and 1xK kernels to represent 2D kernels
    '''
    def __init__(self, inplanes, planes, ks=7):
        super(GCN, self).__init__()
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, ks//2))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(ks//2, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class Refine(nn.Module):
    '''
    Simple residual block to refine the details of the activation maps
    '''
    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)

        out = residual + x
        return out


class FCN(nn.Module):
    '''
    2D FCN network with 3 input channels. The small decoder is built
    with the GCN and Refine modules.
    '''
    def __init__(self, nout=1):
        super(FCN, self).__init__()

        self.nout = nout

        resnet = models.resnet50(pretrained=True)

        self.conv1 = resnet.conv1
        self.bn0 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.gcn1 = GCN(2048, self.nout)
        self.gcn2 = GCN(1024, self.nout)
        self.gcn3 = GCN(512, self.nout)
        self.gcn4 = GCN(64, self.nout)
        self.gcn5 = GCN(64, self.nout)

        self.refine1 = Refine(self.nout)
        self.refine2 = Refine(self.nout)
        self.refine3 = Refine(self.nout)
        self.refine4 = Refine(self.nout)
        self.refine5 = Refine(self.nout)
        self.refine6 = Refine(self.nout)
        self.refine7 = Refine(self.nout)
        self.refine8 = Refine(self.nout)
        self.refine9 = Refine(self.nout)
        self.refine10 = Refine(self.nout)
        self.transformer = nn.Conv2d(256, 64, kernel_size=1)

    def _regresser(self, inplanes):
        return nn.Sequential(
            nn.Conv2d(inplanes, inplanes, 3, padding=1, bias=False),
            nn.BatchNorm2d(inplanes//2),
            nn.ReLU(inplace=True),
            nn.Dropout(.1),
            nn.Conv2d(inplanes//2, self.nout, 1),
        )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn0(x)
        x = self.relu(x)
        conv_x = x
        x = self.maxpool(x)
        pool_x = x

        fm1 = self.layer1(x)
        fm2 = self.layer2(fm1)
        fm3 = self.layer3(fm2)
        fm4 = self.layer4(fm3)

        gcfm1 = self.refine1(self.gcn1(fm4))
        gcfm2 = self.refine2(self.gcn2(fm3))
        gcfm3 = self.refine3(self.gcn3(fm2))
        gcfm4 = self.refine4(self.gcn4(pool_x))
        gcfm5 = self.refine5(self.gcn5(conv_x))

        fs1 = self.refine6(F.upsample_bilinear(gcfm1, fm3.size()[2:]) + gcfm2)
        fs2 = self.refine7(F.upsample_bilinear(fs1, fm2.size()[2:]) + gcfm3)
        fs3 = self.refine8(F.upsample_bilinear(fs2, pool_x.size()[2:]) + gcfm4)
        fs4 = self.refine9(F.upsample_bilinear(fs3, conv_x.size()[2:]) + gcfm5)
        out = self.refine10(F.upsample_bilinear(fs4, input.size()[2:]))

        return out


if __name__ == "__main__":

    # enable_deep_supervision = True
    # model = create_nnunet_from_plans(nnunet_plans, 1, 1, enable_deep_supervision)
    # input = torch.randn(1, 1, 160, 224, 96)
    # output = model(input)
    # if enable_deep_supervision:
    #     for i in range(len(output)):
    #         print(output[i].shape)
    # else:
    #     print(output.shape)

    # print(output.shape)

    model2d = FCN()
    net = AHNet(num_classes=1)
    net.copy_from(model2d)
    input = torch.randn(1, 1, 128, 192, 160)
    output = net(input)
    print(output.shape)