
import numpy as np
from typing import Dict, Hashable, Mapping
from scipy.ndimage.morphology import binary_erosion
import scipy.ndimage as ndi
import torch
import monai.transforms as transforms
from monai.config import KeysCollection
from monai.transforms import MapTransform
import torchio as tio

rs = np.random.RandomState()


class SpinalCordContourd(MapTransform):
    def __init__(
        self,
        keys: KeysCollection,
        allow_missing_keys: bool = False,
    ) -> None:
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        d = dict(data)

        for key in self.keys:
            d[key] = self.create_contour_mask_3d(d[key])
                
        return d

    def create_contour_mask_3d(self, segmentation_mask):
        # Get the shape of the 3D mask
        depth = segmentation_mask.shape[-1]
        
        # Initialize the contour mask
        contour_mask = torch.zeros_like(segmentation_mask)
        
        # Process each slice
        for i in range(depth):
            # Extract the 2D slice
            slice_2d = segmentation_mask[0, :, :, i]

            # Skip the slice if it is empty (because of padding)
            if torch.sum(slice_2d) == 0:
                continue
            
            # Ensure the slice is binary
            binary_slice = (slice_2d > 0).astype(torch.uint8)
            
            # Perform binary erosion
            # eroded_slice = binary_erosion(binary_slice, structure=kernel).astype(np.uint8)
            eroded_slice = binary_erosion(binary_slice)
            
            # Subtract the eroded image from the original to get the contour
            contour_slice = binary_slice - eroded_slice
            
            # Store the contour slice in the contour mask
            contour_mask[0, :, :, i] = contour_slice
        
        return contour_mask

    def inverse(self, data: Mapping[Hashable, torch.Tensor]) -> Dict[Hashable, torch.Tensor]:
        return data


def train_transforms_totalspineseg(crop_size, lbl_key="label", pad_mode="zero", device="cuda"):
    
    transforms_monai = [
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
        # Contrast augmentation
        transforms.RandLambdad(keys=["image"], func=lambda x: ndi.laplace(x)),                          # laplacian
        transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.3),    # this is monai's RandomGamma
        transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        transforms.HistogramNormalized(keys=["image"], num_bins=256, min=0.0, max=1.0),
        transforms.RandLambdad(keys=["image"], func=lambda x: torch.log(1 + x), prob=0.05),             # log
        transforms.RandLambdad(keys=["image"], func=lambda x: torch.sqrt(x), prob=0.05),                # square root
        transforms.RandLambdad(keys=["image"], func=lambda x: torch.exp(x), prob=0.05),                 # exponential
        transforms.RandLambdad(keys=["image"], func=lambda x: torch.sin(x), prob=0.05),                 # sine
        transforms.RandLambdad(keys=["image"], func=lambda x: 1/(1+torch.exp(-x)), prob=0.05),          # sigmoid
        # transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.1),                  # nnUNet's BrightnessMultiplicativeTransform
        # transforms.RandGaussianSharpen(keys=["image"], prob=0.1),
    ]
    # todo: add inverse color augmentation

    # artifacts augmentation
    if rs.rand() < 0.7:
        transforms_monai.append(rs.choice([
            tio.RandomMotion(include=["image", lbl_key]), 
            tio.RandomGhosting(include=["image", lbl_key]),
            tio.RandomSpike(intensity=(1,2), include=["image"]),
            tio.RandomBiasField(include=["image"]),
            tio.RandomBlur(include=["image"]),
        ]))
    transforms_monai.append(transforms.RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1))
    transforms_monai.append(transforms.RandGaussianSharpend(keys=["image"], prob=0.1),)

    # spatial augmentation
    transforms_monai.append(tio.RandomFlip(axes=('LR'), flip_probability=0.3, include=["image", lbl_key]))

    if rs.rand() < 0.7:
        transforms_monai.append(rs.choice([
            tio.RandomAffine(image_interpolation='bspline', label_interpolation='linear', include=["image", lbl_key]),
            tio.RandomAffine(image_interpolation='linear', label_interpolation='nearest', include=["image", lbl_key]),
            tio.RandomElasticDeformation(max_displacement=30, include=["image", lbl_key]),
        ]))

    # simulate low resolution
    if rs.rand() < 0.7:
        transforms_monai.append(tio.RandomAnisotropy(downsampling=(1.5, 5), include=["image", lbl_key]))

    transforms_monai.append(tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0.5, 99.5), include=["image"]))

    return transforms.Compose(transforms_monai)


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
                       sigma_range=(3.5, 5.5), magnitude_range=(25., 35.),),
                    #    mode=(2, 1), padding_mode="border",),
        transforms.RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
        transforms.RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
        transforms.RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
        transforms.RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
        transforms.RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
        transforms.RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
        transforms.RandFlipd(keys=["image", lbl_key], prob=0.3,),
        transforms.NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        # # select one of: spinal cord contour transform or Identity transform (i.e. no transform)
        # transforms.OneOf(
        #     transforms=[SpinalCordContourd(keys=["label"]), transforms.Identityd(keys=["label"])],
        #     weights=[0.25, 0.75]
        # )
    ]

    return transforms.Compose(monai_transforms)

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
