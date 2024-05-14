import pydoc
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from loguru import logger
from collections import OrderedDict

# ---------------------------- Imports for nnUNet's Model -----------------------------
from batchgenerators.utilities.file_and_folder_operations import join
from utils import recursive_find_python_class

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
def create_nnunet_from_plans(plans, input_channels, output_channels, allow_init = True, 
                             deep_supervision: bool = True):
    """
    Adapted from nnUNet's source code:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/utilities/get_network_from_plans.py#L9
    """

    network_class = plans["arch_class_name"]
    # only the keys that "could" depend on the dataset are defined in main.py
    architecture_kwargs = dict(**plans["arch_kwargs"])
    # rest of the default keys are defined here
    architecture_kwargs.update({
        "kernel_sizes": [
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3], 
            [3, 3, 3],
        ],
        "conv_op": "torch.nn.modules.conv.Conv3d",
        "conv_bias": True,
        "norm_op": "torch.nn.modules.instancenorm.InstanceNorm3d",
        "norm_op_kwargs": {
            "eps": 1e-05,
            "affine": True
        },
        "dropout_op": None,
        "dropout_op_kwargs": None,
        "nonlin": "torch.nn.LeakyReLU",
        "nonlin_kwargs": {"inplace": True}
    })

    for ri in plans["arch_kwargs_requires_import"]:
        if architecture_kwargs[ri] is not None:
            architecture_kwargs[ri] = pydoc.locate(architecture_kwargs[ri])

    nw_class = pydoc.locate(network_class)
    # sometimes things move around, this makes it so that we can at least recover some of that
    if nw_class is None:
        warnings.warn(f'Network class {network_class} not found. Attempting to locate it within '
                      f'dynamic_network_architectures.architectures...')
        
        import dynamic_network_architectures
        
        nw_class = recursive_find_python_class(join(dynamic_network_architectures.__path__[0], "architectures"),
                                               network_class.split(".")[-1],
                                               'dynamic_network_architectures.architectures')
        if nw_class is not None:
            print(f'FOUND IT: {nw_class}')
        else:
            raise ImportError('Network class could not be found, please check/correct your plans file')

    if deep_supervision is not None and 'deep_supervision' not in architecture_kwargs.keys():
        architecture_kwargs['deep_supervision'] = deep_supervision

    network = nw_class(
        input_channels=input_channels,
        num_classes=output_channels,
        **architecture_kwargs
    )

    if hasattr(network, 'initialize') and allow_init:
        network.apply(network.initialize)

    return network


def load_pretrained_swinunetr(model, path_pretrained_weights: str):

    logger.info(f"Loading Weights from the Path {path_pretrained_weights}")
    ssl_dict = torch.load(path_pretrained_weights)
    ssl_weights = ssl_dict["model"]

    # Generate new state dict so it can be loaded to MONAI SwinUNETR Model
    monai_loadable_state_dict = OrderedDict()
    model_prior_dict = model.state_dict()
    model_update_dict = model_prior_dict

    del ssl_weights["encoder.mask_token"]
    del ssl_weights["encoder.norm.weight"]
    del ssl_weights["encoder.norm.bias"]
    del ssl_weights["out.conv.conv.weight"]
    del ssl_weights["out.conv.conv.bias"]

    # this is replacing the encoder. with swinViT. in the keys
    for key, value in ssl_weights.items():
        if key[:8] == "encoder.":
            if key[8:19] == "patch_embed":
                new_key = "swinViT." + key[8:]
            else:
                new_key = "swinViT." + key[8:18] + key[20:]
            monai_loadable_state_dict[new_key] = value
        else:
            monai_loadable_state_dict[key] = value

    model_update_dict.update(monai_loadable_state_dict)
    model.load_state_dict(model_update_dict, strict=True)
    model_final_loaded_dict = model.state_dict()

    # Safeguard test to ensure that weights got loaded successfully
    layer_counter = 0
    for k, _v in model_final_loaded_dict.items():
        if k in model_prior_dict:
            layer_counter = layer_counter + 1

            old_wts = model_prior_dict[k]
            new_wts = model_final_loaded_dict[k]

            old_wts = old_wts.to("cpu").numpy()
            new_wts = new_wts.to("cpu").numpy()
            diff = np.mean(np.abs(old_wts, new_wts))
            logger.info(f"Layer {k}, the update difference is: {diff}")
            if diff == 0.0:
                logger.info(f"Warning: No difference found for layer {k}")
    
    logger.info(f"Total updated layers {layer_counter} / {len(model_prior_dict)}")
    logger.info(f"Pretrained Weights Succesfully Loaded !")

    return model

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
