
import numpy as np
import monai.transforms as transforms
import batchgenerators.transforms.spatial_transforms as bg_spatial_transforms


def train_transforms(crop_size, lbl_key="label", pad_mode="zero", device="cuda"):

    monai_transforms = [
        # pre-processing
        transforms.LoadImaged(keys=["image", lbl_key]),
        transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
        transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
        # NOTE: spine interpolation with order=2 is spline, order=1 is linear
        transforms.Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
        transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,
                                        mode="constant" if pad_mode == "zero" else pad_mode),
        # convert the data to Tensor without meta, move to GPU and cache it to avoid CPU -> GPU sync in every epoch
        transforms.EnsureTyped(keys=["image", lbl_key], device=device, track_meta=False),
        # data-augmentation
        transforms.RandAffined(keys=["image", lbl_key], mode=(2, 1), prob=0.9,
                    rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians
                    scale_range=(-0.2, 0.2),
                    translate_range=(-0.1, 0.1)),
        transforms.Rand3DElasticd(keys=["image", lbl_key], prob=0.5,
                       sigma_range=(3.5, 5.5), magnitude_range=(25., 35.),
                       mode=(2, 1), padding_mode="border",),
        transforms.RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.5),
        transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
        transforms.RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
        transforms.RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
        transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.5),
        transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.35),  # this is nnUNet's BrightnessMultiplicativeTransform
        transforms.RandFlipd(keys=["image", lbl_key], prob=0.5,),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
    ]

    batchgenerators_transforms = [
        bg_spatial_transforms.ChannelTranslation(
            data_key="image",
            const_channel=5,
            max_shifts={'x': 5, 'y': 5, 'z': 5})
    ]

    # add batchgenerators transforms
    transforms_final = monai_transforms + [
        # add another dim as BatchGenerator expects shape [B, C, H, W, D]
        transforms.EnsureChannelFirstd(keys=["image", lbl_key], channel_dim="no_channel"),
        # batchgenerators transforms work on numpy arrays
        transforms.ToNumpyd(keys=["image", lbl_key]),
        # use adaptors to port batchgenerators transforms to monai-compatible transforms
        transforms.adaptor(batchgenerators_transforms[0], {"image": "image", "label": f"{lbl_key}"}),
        # convert the data back to Tensor
        transforms.EnsureTyped(keys=["image", lbl_key], device=device, track_meta=False),
        transforms.SqueezeDimd(keys=[f"{lbl_key}"], dim=0),
    ]

    return transforms.Compose(transforms_final)

def inference_transforms(crop_size, lbl_key="label"):
    return transforms.Compose([
            transforms.LoadImaged(keys=["image", lbl_key], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            transforms.Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            transforms.DivisiblePadd(keys=["image", lbl_key], k=2**5),   # pad inputs to ensure divisibility by no. of layers nnUNet has (5)
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms(crop_size, lbl_key="label", pad_mode="zero"):
    return transforms.Compose([
            transforms.LoadImaged(keys=["image", lbl_key], image_only=False),
            transforms.EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            transforms.Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            transforms.Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            transforms.ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,
                                            mode="constant" if pad_mode == "zero" else pad_mode),
            transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])
