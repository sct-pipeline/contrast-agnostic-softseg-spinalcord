
import numpy as np
from monai.transforms import (SpatialPadd, Compose, CropForegroundd, LoadImaged, RandFlipd, 
            RandCropByPosNegLabeld, Spacingd, RandRotated, NormalizeIntensityd,
            RandWeightedCropd, RandAdjustContrastd, EnsureChannelFirstd, RandGaussianNoised, 
            RandGaussianSmoothd, Orientationd, Rand3DElasticd, RandBiasFieldd)

# median image size in voxels - taken from nnUNet
# median_size = (123, 255, 214)     # so pad with this size
# median_size after 1mm isotropic resampling
# median_size = [ 192. 228. 106.]

# Order in which nnunet does preprocessing:
# 1. Crop to non-zero
# 2. Normalization
# 3. Resample to target spacing

def train_transforms(crop_size, num_samples_pv, lbl_key="label"):
    return Compose([   
            # pre-processing
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear"),),
            # data-augmentation
            SpatialPadd(keys=["image", lbl_key], spatial_size=(192, 228, 106), method="symmetric"),
            # NOTE: used with neg together to calculate the ratio pos / (pos + neg) for the probability to pick a 
            # foreground voxel as a center rather than a background voxel.
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                   spatial_size=crop_size, pos=3, neg=1, num_samples=num_samples_pv, 
                                   # if num_samples=4, then 4 samples/image are randomly generated
                                   image_key="image", image_threshold=0.),
            # RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
            Rand3DElasticd(keys=["image", lbl_key], sigma_range=(3.5, 5.5), magnitude_range=(25, 35), prob=0.5),    
            # TODO: Try Spacingd with low resolution here with prob=0.5
            RandAdjustContrastd(keys=["image"], gamma=(0.7, 1.5), prob=0.4),    # this is monai's RandomGamma
            RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
            RandGaussianSmoothd(keys=["image"], sigma_x=(0.0, 2.0), sigma_y=(0.0, 2.0), sigma_z=(0.0, 2.0), prob=0.3),
            RandFlipd(keys=["image", lbl_key], spatial_axis=None, prob=0.4,),
            RandRotated(keys=["image", lbl_key], mode=("bilinear", "nearest"), prob=0.2,
                        range_x=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),  # NOTE: -pi/6 to pi/6
                        range_y=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi), 
                        range_z=(-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi)),
            # # re-orientation
            # Orientationd(keys=["image", lbl_key], axcodes="RPI"),   # NOTE: if not using it here, then it results in collation error
        ])

def val_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear"),),
            # SpatialPadd(keys=["image", lbl_key], spatial_size=(123, 255, 214), method="symmetric"),
        ])
