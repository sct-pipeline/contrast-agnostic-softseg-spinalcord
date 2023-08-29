"""
Some useful blocks for building the network architecture.
"""

import torch.nn as nn


def conv_norm_lrelu(feat_in, feat_out):
    """Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def norm_lrelu_conv(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def lrelu_conv(feat_in, feat_out):
    """LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def norm_lrelu_upscale_conv_norm_lrelu(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + 2X Upsample + Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


class InitWeights_He(object):
    def __init__(self, neg_slope=1e-2):
        self.neg_slope = neg_slope

    def __call__(self, module):
        if isinstance(module, nn.Conv3d) or isinstance(module, nn.ConvTranspose3d):
            module.weight = nn.init.kaiming_normal_(module.weight, a=self.neg_slope)
            if module.bias is not None:
                module.bias = nn.init.constant_(module.bias, 0)
