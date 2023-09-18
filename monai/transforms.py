
import numpy as np
from monai.transforms import (SpatialPadd, Compose, CropForegroundd, LoadImaged, RandFlipd, 
            RandCropByPosNegLabeld, Spacingd, RandScaleIntensityd, NormalizeIntensityd, RandAffined,
            RandWeightedCropd, RandAdjustContrastd, EnsureChannelFirstd, RandGaussianNoised, 
            RandGaussianSmoothd, Orientationd, Rand3DElasticd, RandBiasFieldd, RandSimulateLowResolutiond,
            ResizeWithPadOrCropd)

# median image size in voxels - taken from nnUNet
# median_size = (123, 255, 214)  as per 0.9 iso resampling and patch_size = (80, 192, 160)
# note the the order of the axes is different in nnunet and monai (dims 0 and 2 are swapped)
# median_size after 1mm isotropic resampling
# median_size = [ 192. 228. 106.]   

# Order in which nnunet does preprocessing:
# 1. Crop to non-zero
# 2. Normalization
# 3. Resample to target spacing

# Order in which ivadomed does preprocessing:
# 1. Resample to 1mm iso
# 2. CenterCrop using 46x176x288
# 3. RandomAffine --> RandomElastic --> RandomGamma --> RandomBiasField --> RandomBlur --> NormalizeInstance


def train_transforms(crop_size, num_samples_pv, lbl_key="label"):

    monai_transforms = [    
        # pre-processing
        LoadImaged(keys=["image", lbl_key]),
        EnsureChannelFirstd(keys=["image", lbl_key]),
        CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
        # NOTE: spine interpolation with order=2 is spline, order=1 is linear
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
        # data-augmentation
        # SpatialPadd(keys=["image", lbl_key], spatial_size=(192, 228, 106), method="symmetric"),
        SpatialPadd(keys=["image", lbl_key], spatial_size=crop_size, method="symmetric"),   # pad with the same size as crop_size
        # NOTE: used with neg together to calculate the ratio pos / (pos + neg) for the probability to pick a 
        # foreground voxel as a center rather than a background voxel.
        RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                            spatial_size=crop_size, pos=3, neg=1, num_samples=num_samples_pv, 
                            # if num_samples=4, then 4 samples/image are randomly generated
                            image_key="image", image_threshold=0.),
        # re-ordering transforms as used by nnunet
        RandAffined(keys=["image", lbl_key], mode=(2, 1), prob=0.75,
                    rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians 
                    scale_range=(-0.2, 0.2),   
                    translate_range=(-0.1, 0.1)),
        Rand3DElasticd(keys=["image", lbl_key], prob=0.5,
                       sigma_range=(3.5, 5.5), 
                       magnitude_range=(25., 35.)),
        RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
        RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
        RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
        RandGaussianNoised(keys=["image"], mean=0.0, std=0.1, prob=0.1),
        RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
        RandScaleIntensityd(keys=["image"], factors=(-0.25, 1), prob=0.15),  # this is nnUNet's BrightnessMultiplicativeTransform
        RandFlipd(keys=["image", lbl_key], prob=0.3,),
        NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        
        # # defining transforms as used by ivadomed (with the same probabilities)
        # LoadImaged(keys=["image", lbl_key], image_only=False),   # image_only=True to avoid loading the label
        # EnsureChannelFirstd(keys=["image", lbl_key]),
        # Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)),
        # ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
        # RandAffined(keys=["image", lbl_key], mode=(2, 1), prob=1.0,
        #             rotate_range=(-20. / 360 * 2. * np.pi, 20. / 360 * 2. * np.pi),    # monai expects in radians 
        #             scale_range=(-0.2, 0.2),   # ivadomed uses sth like scale_x = random.uniform(1 - self.scale[0], 1 + self.scale[0]), but monai adds 1.0 to the scale
        #             translate_range=(-0.1, 0.1)),
        # Rand3DElasticd(keys=["image", lbl_key], prob=0.5,
        #                sigma_range=(3.5, 5.5), 
        #                magnitude_range=(25., 35.)),
        # # RandSimulateLowResolutiond(keys=["image"], zoom_range=(0.5, 1.0), prob=0.25),
        # RandAdjustContrastd(keys=["image"], gamma=(0.5, 3.), prob=0.5),    # this is monai's RandomGamma
        # RandBiasFieldd(keys=["image"], coeff_range=(0.0, 0.5), degree=3, prob=0.3),
        # RandGaussianSmoothd(keys=["image"], sigma_x=(0., 2.), sigma_y=(0., 2.), sigma_z=(0., 2.0), prob=0.3),
        # # RandFlipd(keys=["image", lbl_key], prob=0.5,),
        # NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
    ]

    return Compose(monai_transforms) 

def val_transforms_without_center_crop(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], image_only=False),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms_with_orientation_and_crop(crop_size, lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], image_only=False),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            # TODO: do cropping only in R-L so sth like (48, -1, -1)
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])

def val_transforms(crop_size, lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key], image_only=False),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Spacingd(keys=["image", lbl_key], pixdim=(1.0, 1.0, 1.0), mode=(2, 1)), # mode=("bilinear", "bilinear"),),
            ResizeWithPadOrCropd(keys=["image", lbl_key], spatial_size=crop_size,),
            # TODO: do cropping only in R-L so sth like (48, -1, -1)
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=False),
        ])
