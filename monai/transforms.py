
from monai.transforms import (SpatialPadd, Compose, CropForegroundd, LoadImaged, RandFlipd, 
            RandCropByPosNegLabeld, Spacingd, RandRotate90d, ToTensord, NormalizeIntensityd, 
            EnsureType, RandWeightedCropd, HistogramNormalized, EnsureTyped, Invertd, SaveImaged,
            EnsureChannelFirstd, CenterSpatialCropd, RandSpatialCropSamplesd, Orientationd)

# median image size in voxels - taken from nnUNet
# median_size = (123, 255, 214)
# so pad with this size

def train_transforms(crop_size, num_samples_pv, lbl_key="label"):
    return Compose([   
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            # Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            # TODO: if the source_key is set to "label", then the cropping is only around the label mask
            CropForegroundd(keys=["image", lbl_key], source_key="image"),     # crops >0 values with a bounding box
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear"),),
            SpatialPadd(keys=["image", lbl_key], spatial_size=(123, 255, 214), method="symmetric"),
            # RandSpatialCropSamplesd(keys=["image", lbl_key], roi_size=crop_size, num_samples=num_samples_pv, random_center=True, random_size=False),
            RandCropByPosNegLabeld(keys=["image", "label"], label_key="label",
                                   spatial_size=crop_size, pos=1, neg=1, num_samples=num_samples_pv, 
                                   # if num_samples=4, then 4 samples/image are randomly generated
                                   image_key="image", image_threshold=0.), 
            RandFlipd(keys=["image", lbl_key], spatial_axis=[0], prob=0.50,),
            RandFlipd(keys=["image", lbl_key], spatial_axis=[1], prob=0.50,),
            RandFlipd(keys=["image", lbl_key],spatial_axis=[2],prob=0.50,),
            RandRotate90d(keys=["image", lbl_key], prob=0.10, max_k=3,),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),   # NOTE: if not using it here, then it results in collation error
            # HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            # ToTensord(keys=["image", lbl_key]), 
        ])

def val_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear"),),
            # SpatialPadd(keys=["image", lbl_key], spatial_size=(123, 255, 214), method="symmetric"),
            # HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            # ToTensord(keys=["image", lbl_key]),
        ])

def test_transforms(lbl_key="label"):
    return Compose([
            LoadImaged(keys=["image", lbl_key]),
            EnsureChannelFirstd(keys=["image", lbl_key]),
            Orientationd(keys=["image", lbl_key], axcodes="RPI"),
            CropForegroundd(keys=["image", lbl_key], source_key="image"),
            Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "bilinear"),),
            # AddChanneld(keys=["image", lbl_key]),
            # HistogramNormalized(keys=["image"], mask=None),
            NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True),
            # ToTensord(keys=["image", lbl_key]),
        ])