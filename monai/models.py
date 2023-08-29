import torch
import torch.nn as nn
import torch.nn.functional as F
from building_blocks import conv_norm_lrelu, norm_lrelu_conv, lrelu_conv, norm_lrelu_upscale_conv_norm_lrelu, InitWeights_He

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


# ---------------------------- ModifiedUNet3D Encoder Implementation -----------------------------
class ModifiedUNet3DEncoder(nn.Module):
    """Encoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, in_channels=1, base_n_filter=8):
        super(ModifiedUNet3DEncoder, self).__init__()

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = lrelu_conv(base_n_filter, base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(base_n_filter, base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = norm_lrelu_conv(base_n_filter * 2, base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(base_n_filter * 2, base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = norm_lrelu_conv(base_n_filter * 4, base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(base_n_filter * 4, base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = norm_lrelu_conv(base_n_filter * 8, base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = norm_lrelu_conv(base_n_filter * 16, base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 16, base_n_filter * 8)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)

        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5

        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        context_features = [context_1, context_2, context_3, context_4]

        return out, context_features


# ---------------------------- ModifiedUNet3D Decoder Implementation -----------------------------
class ModifiedUNet3DDecoder(nn.Module):
    """Decoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, n_classes=1, base_n_filter=8):
        super(ModifiedUNet3DDecoder, self).__init__()

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_l0 = nn.Conv3d(base_n_filter * 8, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = conv_norm_lrelu(base_n_filter * 16, base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(base_n_filter * 16, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 8, base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = conv_norm_lrelu(base_n_filter * 8, base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(base_n_filter * 8, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 4, base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = conv_norm_lrelu(base_n_filter * 2, base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, context_features):
        # Get context features from the encoder
        context_1, context_2, context_3, context_4 = context_features

        out = self.conv3d_l0(x)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsample(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        # # Final Activation Layer
        # out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)
        # out = out.squeeze()

        return out  # this is just the logits, not the probablities


# ---------------------------- ModifiedUNet3D Implementation -----------------------------
class ModifiedUNet3D(nn.Module):
    """ModifiedUNet3D with Encoder + Decoder. Adapted from ivadomed.models"""
    def __init__(self, in_channels=1, out_channels=1, init_filters=8):
        super(ModifiedUNet3D, self).__init__()
        self.unet_encoder = ModifiedUNet3DEncoder(in_channels=in_channels, base_n_filter=init_filters)
        self.unet_decoder = ModifiedUNet3DDecoder(n_classes=out_channels, base_n_filter=init_filters)

    def forward(self, x):

        x, context_features = self.unet_encoder(x)
        # x: (B, 8 * F, SV // 8, SV // 8, SV // 8)
        # context_features: [4]
        #   0 -> (B, F, SV, SV, SV)
        #   1 -> (B, 2 * F, SV / 2, SV / 2, SV / 2)
        #   2 -> (B, 4 * F, SV / 4, SV / 4, SV / 4)
        #   3 -> (B, 8 * F, SV / 8, SV / 8, SV / 8)

        seg_logits = self.unet_decoder(x, context_features)

        return seg_logits
    


if __name__ == "__main__":

    enable_deep_supervision = True
    model = create_nnunet_from_plans(nnunet_plans, 1, 1, enable_deep_supervision)
    input = torch.randn(1, 1, 160, 224, 96)
    output = model(input)
    if enable_deep_supervision:
        for i in range(len(output)):
            print(output[i].shape)
    else:
        print(output.shape)

    # print(output.shape)
