import os
import argparse
from datetime import datetime
from loguru import logger
import yaml
import json
import time

import numpy as np
import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import dice_score, PolyLRScheduler, check_empty_patch, count_parameters, get_datasets_stats
from losses import AdapWingLoss
from transforms import train_transforms, val_transforms
from models import create_nnunet_from_plans, load_pretrained_weights, load_pretrained_swinunetr

from monai.apps import download_url
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNETR, SwinUNETR
from monai.data import (ThreadDataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, set_track_meta)
from monai.transforms import (Compose, EnsureType, EnsureTyped, Invertd, SaveImage)

# list of contrasts and their possible various names in the datasets
CONTRASTS = {
    "t1map": ["T1map"],
    "t1w": ["T1w", "space-other_T1w", "acq-lowresSag_T1w"],
    "t2w": ["T2w", "space-other_T2w", "acq-lowresSag_T2w", "acq-highresSag_T2w"],
    "t2star": ["T2star", "space-other_T2star"],
    "dwi": ["rec-average_dwi", "acq-dwiMean_dwi"],
    "mt-on": ["flip-1_mt-on_space-other_MTS", "acq-MTon_MTR"],
    "mt-off": ["flip-2_mt-off_space-other_MTS"],
    "unit1": ["UNIT1"],
    "psir": ["PSIR"],
    "stir": ["STIR"]
}

def get_args():
    parser = argparse.ArgumentParser(description='Script for training contrast-agnositc SC segmentation model.')

    # arguments for model
    parser.add_argument('-m', '--model', choices=['nnunet-plain', 'nnunet-resencM', 'swinunetr', 'meunet'],
                        default='nnunet', type=str,
                        help='Model type to be used. Options: nnunet, swinunetr.')
    # path to the config file
    parser.add_argument("--config", type=str, default="./config.json",
                        help="Path to the config file containing all training details.")
    # saving
    parser.add_argument('--debug', default=False, action='store_true', help='if true, results are not logged to wandb')
    parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true',
                            help='Load model from checkpoint and continue training')
    # temporaray arg
    parser.add_argument('--pad-mode', type=str, default="zero", help="Padding mode for the images.")
    parser.add_argument('--input-label', type=str, default="soft", choices=["soft", "bin"],
                        help="Type of label input to the model. 'soft' for the SoftSeg approach, 'bin' for binarize labels after SoftSeg (data-aug).")
    # path to the folder containing pre-trained weights
    parser.add_argument('--pretrained-path', type=str, default=None, help="Path to the folder containing pre-trained weights.")

    args = parser.parse_args()

    return args


# create a "model"-agnostic class with PL to use different models
class Model(pl.LightningModule):
    def __init__(self, config, data_root, net, loss_function, optimizer_class, exp_id=None, results_path=None):
        super().__init__()
        self.cfg = config
        self.save_hyperparameters(ignore=['net', 'loss_function'])

        self.input_label_type = args.input_label
        self.root = data_root
        self.net = net
        self.lr = config["opt"]["lr"]
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.save_exp_id = exp_id
        self.results_path = results_path

        self.best_val_dice, self.best_val_epoch = 0, 0
        self.best_val_loss = float("inf")

        # define cropping and padding dimensions
        # NOTE about patch sizes: nnUNet defines patches using the median size of the dataset as the reference
        # BUT, for SC images, this means a lot of context outside the spinal cord is included in the patches
        # which could be sub-optimal.
        # On the other hand, ivadomed used a patch-size that's heavily padded along the R-L direction so that
        # only the SC is in context.
        self.spacing = config["preprocessing"]["spacing"]
        self.voxel_cropping_size = self.inference_roi_size = config["preprocessing"]["crop_pad_size"]

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = self.val_post_label = Compose([EnsureType(track_meta=False)])

        # define evaluation metric
        self.soft_dice_metric = dice_score

        # temp lists for storing outputs from training, validation, and testing
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []


    # --------------------------------
    # FORWARD PASS
    # --------------------------------
    def forward(self, x):

        out = self.net(x)
        # # NOTE: MONAI's models only output the logits, not the output after the final activation function
        # # https://docs.monai.io/en/0.9.0/_modules/monai/networks/nets/unetr.html#UNETR.forward refers to the
        # # UnetOutBlock (https://docs.monai.io/en/0.9.0/_modules/monai/networks/blocks/dynunet_block.html#UnetOutBlock)
        # # as the final block applied to the input, which is just a convolutional layer with no activation function
        # # Hence, we are used Normalized ReLU to normalize the logits to the final output
        # normalized_out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)

        return out  # returns logits


    # --------------------------------
    # DATA PREPARATION
    # --------------------------------
    def prepare_data(self):
        # set deterministic training for reproducibility
        set_determinism(seed=self.cfg["seed"])

        # define training and validation transforms
        transforms_train = train_transforms(
            crop_size=self.voxel_cropping_size,
            lbl_key='label',
            pad_mode=args.pad_mode,
            device=self.device,
        )
        transforms_val = val_transforms(crop_size=self.inference_roi_size, lbl_key='label', pad_mode=args.pad_mode,)

        # get the dataset statistics
        save_path = os.path.join(self.cfg["directories"]["models_dir"], os.path.basename(self.results_path))
        get_datasets_stats(datalists_root=self.root, contrasts_dict=CONTRASTS, path_save=save_path)
        logger.info(f"Dataset statistics saved to {save_path} ...")

        # get all datalists
        datalists_list = [f for f in os.listdir(self.root) if f.endswith("_seed50.json")]
        # load the dataset
        train_files, val_files, test_files = [], [], []
        for datalist in datalists_list:
            logger.info(f"Loading dataset: {datalist}")
            train_files += load_decathlon_datalist(os.path.join(self.root, datalist), True, "train")
            val_files += load_decathlon_datalist(os.path.join(self.root, datalist), True, "validation")
            test_files += load_decathlon_datalist(os.path.join(self.root, datalist), True, "test")

        logger.info(f"Combining {len(datalists_list)} datasets ...")
        logger.info(f"Number of training samples (i.e. images, not subjects): {len(train_files)}")
        logger.info(f"Number of validation samples (i.e. images, not subjects): {len(val_files)}")
        logger.info(f"Number of testing samples (i.e. images, not subjects): {len(test_files)}")

        if args.debug:
            train_files = train_files[:50]
            val_files = val_files[:15]
            test_files = test_files[:6]

        train_cache_rate = 0.5 # 0.25 if args.model == 'swinunetr' else 0.5
        self.train_ds = CacheDataset(data=train_files, transform=transforms_train, cache_rate=train_cache_rate, num_workers=12,
                                     copy_cache=False)
        self.val_ds = CacheDataset(data=val_files, transform=transforms_val, cache_rate=0.25, num_workers=12,
                                   copy_cache=False)

        # define test transforms
        transforms_test = val_transforms(crop_size=self.inference_roi_size, lbl_key='label', pad_mode=args.pad_mode,)

        # define post-processing transforms for testing; taken (with explanations) from
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        self.test_post_pred = Compose([
            EnsureTyped(keys=["pred", "label"]),
            Invertd(keys=["pred", "label"], transform=transforms_test,
                    orig_keys=["image", "label"],
                    meta_keys=["pred_meta_dict", "label_meta_dict"],
                    nearest_interp=False, to_tensor=True),
            ])
        self.test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.1, num_workers=8,
                                    copy_cache=False)

        # # avoid the computation of meta information in random transforms
        # set_track_meta(False)

    # --------------------------------
    # DATA LOADERS
    # --------------------------------
    def train_dataloader(self):
        return ThreadDataLoader(self.train_ds, batch_size=self.cfg["opt"]["batch_size"], shuffle=True, num_workers=8,
                            pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return ThreadDataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                          persistent_workers=True)

    def test_dataloader(self):
        return ThreadDataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)


    # --------------------------------
    # OPTIMIZATION
    # --------------------------------
    def configure_optimizers(self):
        if self.cfg["opt"]["name"] == "sgd":
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, momentum=0.99, weight_decay=3e-5, nesterov=True)
            scheduler = PolyLRScheduler(optimizer, self.lr, max_steps=self.cfg["opt"]["max_epochs"])
        else:
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, fused=True)
            # NOTE: ivadomed using CosineAnnealingLR with T_max = 200
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg["opt"]["max_epochs"])
        return [optimizer], [scheduler]


    # --------------------------------
    # TRAINING
    # --------------------------------
    def training_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["label"]

        if self.input_label_type == "bin":
            # binarize the labels with a threshold of 0.5
            # NOTE: if "bin" is used, then it is not SoftSeg approach
            labels = (labels > 0.5).float()

        # check if any label image patch is empty in the batch
        if check_empty_patch(labels) is None:
            # print(f"Empty label patch found. Skipping training step ...")
            return None

        output = self.forward(inputs)   # logits
        # print(f"labels.shape: {labels.shape} \t output.shape: {output.shape}")

        if args.model in ["nnunet-plain", "nnunet-resencM"] and self.cfg['model'][args.model]["enable_deep_supervision"]:

            # calculate dice loss for each output
            loss, train_soft_dice = 0.0, 0.0
            for i in range(len(output)):
                # give each output a weight which decreases exponentially (division by 2) as the resolution decreases
                # this gives higher resolution outputs more weight in the loss
                # NOTE: outputs[0] is the final pred, outputs[-1] is the lowest resolution pred (at the bottleneck)
                # we're downsampling the GT to the resolution of each deepsupervision feature map output
                # (instead of upsampling each deepsupervision feature map output to the final resolution)
                downsampled_gt = F.interpolate(labels, size=output[i].shape[-3:], mode='trilinear', align_corners=False)
                # print(f"downsampled_gt.shape: {downsampled_gt.shape} \t output[i].shape: {output[i].shape}")
                loss += (0.5 ** i) * self.loss_function(output[i], downsampled_gt)

                # get probabilities from logits
                out = F.relu(output[i]) / F.relu(output[i]).max() if bool(F.relu(output[i]).max()) else F.relu(output[i])

                # calculate train dice
                # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
                # So, take this dice score with a lot of salt
                train_soft_dice += self.soft_dice_metric(out, downsampled_gt)

            # average dice loss across the outputs
            loss /= len(output)
            train_soft_dice /= len(output)

        else:
            # calculate training loss
            loss = self.loss_function(output, labels)

            # get probabilities from logits
            output = F.relu(output) / F.relu(output).max() if bool(F.relu(output).max()) else F.relu(output)

            # calculate train dice
            # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
            # So, take this dice score with a lot of salt
            train_soft_dice = self.soft_dice_metric(output, labels)

        metrics_dict = {
            "loss": loss.cpu(),
            "train_soft_dice": train_soft_dice.detach().cpu(),
            "train_number": len(inputs),
            # "train_image": inputs[0].detach().cpu().squeeze(),
            # "train_gt": labels[0].detach().cpu().squeeze(),
            # "train_pred": output[0].detach().cpu().squeeze()
        }
        self.train_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_train_epoch_end(self):

        if self.train_step_outputs == []:
            # means the training step was skipped because of empty input patch
            return None
        else:
            train_loss, train_soft_dice = 0, 0
            num_items = len(self.train_step_outputs)
            for output in self.train_step_outputs:
                train_loss += output["loss"].item()
                train_soft_dice += output["train_soft_dice"].item()

            mean_train_loss = (train_loss / num_items)
            mean_train_soft_dice = (train_soft_dice / num_items)

            wandb_logs = {
                "train_soft_dice": mean_train_soft_dice,
                "train_loss": mean_train_loss,
            }
            self.log_dict(wandb_logs)

            # # plot the training images
            # fig = plot_slices(image=self.train_step_outputs[0]["train_image"],
            #                   gt=self.train_step_outputs[0]["train_gt"],
            #                   pred=self.train_step_outputs[0]["train_pred"],
            #                   debug=args.debug)
            # wandb.log({"training images": wandb.Image(fig)})

            # free up memory
            self.train_step_outputs.clear()
            wandb_logs.clear()
            # plt.close(fig)


    # --------------------------------
    # VALIDATION
    # --------------------------------
    def validation_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["label"]

        if self.input_label_type == "bin":
            # binarize the labels with a threshold of 0.5
            labels = (labels > 0.5).float()

        # NOTE: this calculates the loss on the entire image after sliding window
        outputs = sliding_window_inference(inputs, self.inference_roi_size, mode="gaussian",
                                           sw_batch_size=4, predictor=self.forward, overlap=0.5,)
        # outputs shape: (B, C, <original H x W x D>)
        if args.model in ["nnunet-plain", "nnunet-resencM"] and self.cfg['model'][args.model]["enable_deep_supervision"]:
            # we only need the output with the highest resolution
            outputs = outputs[0]

        # calculate validation loss
        loss = self.loss_function(outputs, labels)

        # get probabilities from logits
        outputs = F.relu(outputs) / F.relu(outputs).max() if bool(F.relu(outputs).max()) else F.relu(outputs)

        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]
        val_soft_dice = self.soft_dice_metric(post_outputs[0], post_labels[0])

        hard_preds, hard_labels = (post_outputs[0].detach() > 0.5).float(), (post_labels[0].detach() > 0.5).float()
        val_hard_dice = self.soft_dice_metric(hard_preds, hard_labels)

        # NOTE: there was a massive memory leak when storing cuda tensors in this dict. Hence,
        # using .detach() to avoid storing the whole computation graph
        # Ref: https://discuss.pytorch.org/t/cuda-memory-leak-while-training/82855/2
        metrics_dict = {
            "val_loss": loss.detach().cpu(),
            "val_soft_dice": val_soft_dice.detach().cpu(),
            "val_hard_dice": val_hard_dice.detach().cpu(),
            "val_number": len(post_outputs),
            # "val_image": inputs[0].detach().cpu().squeeze(),
            # "val_gt": labels[0].detach().cpu().squeeze(),
            # "val_pred": post_outputs[0].detach().cpu().squeeze(),
        }
        self.val_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_validation_epoch_end(self):

        val_loss, num_items, val_soft_dice, val_hard_dice = 0, 0, 0, 0
        for output in self.val_step_outputs:
            val_loss += output["val_loss"].sum().item()
            val_soft_dice += output["val_soft_dice"].sum().item()
            val_hard_dice += output["val_hard_dice"].sum().item()
            num_items += output["val_number"]

        mean_val_loss = (val_loss / num_items)
        mean_val_soft_dice = (val_soft_dice / num_items)
        mean_val_hard_dice = (val_hard_dice / num_items)

        wandb_logs = {
            "val_soft_dice": mean_val_soft_dice,
            "val_hard_dice": mean_val_hard_dice,
            "val_loss": mean_val_loss,
        }
        # save the best model based on validation dice score
        if mean_val_soft_dice > self.best_val_dice:
            self.best_val_dice = mean_val_soft_dice
            self.best_val_epoch = self.current_epoch

        # save the best model based on validation CSA loss
        if mean_val_loss < self.best_val_loss:
            self.best_val_loss = mean_val_loss
            self.best_val_epoch = self.current_epoch

        logger.info(
            f"\nCurrent epoch: {self.current_epoch}"
            f"\nAverage Soft Dice (VAL): {mean_val_soft_dice:.4f}"
            f"\nAverage Hard Dice (VAL): {mean_val_hard_dice:.4f}"
            f"\nAverage AdapWing Loss (VAL): {mean_val_loss:.4f}"
            f"\nBest Average AdapWing Loss: {self.best_val_loss:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")


        # log on to wandb
        self.log_dict(wandb_logs)

        # # plot the validation images
        # fig = plot_slices(image=self.val_step_outputs[0]["val_image"],
        #                   gt=self.val_step_outputs[0]["val_gt"],
        #                   pred=self.val_step_outputs[0]["val_pred"],)
        # wandb.log({"validation images": wandb.Image(fig)})

        # free up memory
        self.val_step_outputs.clear()
        wandb_logs.clear()
        # plt.close(fig)

        # return {"log": wandb_logs}

    # --------------------------------
    # TESTING
    # --------------------------------
    def test_step(self, batch, batch_idx):

        test_input = batch["image"]
        # print(batch["label_meta_dict"]["filename_or_obj"][0])
        batch["pred"] = sliding_window_inference(test_input, self.inference_roi_size,
                                                 sw_batch_size=4, predictor=self.forward, overlap=0.5)

        if args.model in ["nnunet-plain", "nnunet-resencM"] and self.cfg['model'][args.model]["enable_deep_supervision"]:
            # we only need the output with the highest resolution
            batch["pred"] = batch["pred"][0]

        # normalize the logits
        batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() if bool(F.relu(batch["pred"]).max()) else F.relu(batch["pred"])

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]

        # make sure that the shapes of prediction and GT label are the same
        # print(f"pred shape: {post_test_out[0]['pred'].shape}, label shape: {post_test_out[0]['label'].shape}")
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape

        pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

        # save the prediction and label
        if self.cfg["save_test_preds"]:

            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            logger.info(f"Saving subject: {subject_name}")

            # image saver class
            save_folder = os.path.join(self.results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz",
                separate_folder=False, print_log=False, resample=True)
            # save the prediction
            pred_saver(pred)

            # label_saver = SaveImage(
            #     output_dir=save_folder, output_postfix="gt", output_ext=".nii.gz",
            #     separate_folder=False, print_log=False, resample=True)
            # # save the label
            # label_saver(label)


        # NOTE: Important point from the SoftSeg paper - binarize predictions before computing metrics
        # calculate soft and hard dice here (for quick overview), other metrics can be computed from
        # the saved predictions using ANIMA
        # 1. Dice Score
        test_soft_dice = self.soft_dice_metric(pred, label)

        # binarizing the predictions
        pred = (post_test_out[0]['pred'].detach().cpu() > 0.5).float()
        label = (post_test_out[0]['label'].detach().cpu() > 0.5).float()

        # 1.1 Hard Dice Score
        test_hard_dice = self.soft_dice_metric(pred.numpy(), label.numpy())

        metrics_dict = {
            "test_hard_dice": test_hard_dice,
            "test_soft_dice": test_soft_dice,
        }
        self.test_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_test_epoch_end(self):

        avg_hard_dice_test, std_hard_dice_test = np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).std()
        avg_soft_dice_test, std_soft_dice_test = np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).std()

        logger.info(f"Test (Soft) Dice: {avg_soft_dice_test}")
        logger.info(f"Test (Hard) Dice: {avg_hard_dice_test}")

        self.avg_test_dice, self.std_test_dice = avg_soft_dice_test, std_soft_dice_test
        self.avg_test_dice_hard, self.std_test_dice_hard = avg_hard_dice_test, std_hard_dice_test

        # free up memory
        self.test_step_outputs.clear()


# --------------------------------
# MAIN
# --------------------------------
def main(args):

    # load config file
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Setting the seed
    pl.seed_everything(config["seed"], workers=True)

    # define root path for finding datalists
    dataset_root = config["dataset"]["root_dir"]

    all_contrasts, datasets = [], []
    datalists_list = [os.path.join(dataset_root, f) for f in os.listdir(dataset_root) if f.endswith("_seed50.json")]
    for datalist in datalists_list:
        with open(datalist, "r") as f:
            datalist = json.load(f)
            all_contrasts += datalist["contrasts"]
            datasets.append(datalist["dataset"])

    contrasts_final = []
    for k, v in CONTRASTS.items():
        for c in all_contrasts:
            if c in v:
                contrasts_final.append(k)
                break

    n_contrasts, n_datasets = len(contrasts_final), len(datasets)

    # define optimizer
    if config["opt"]["name"] == "adam":
        optimizer_class = torch.optim.Adam
    elif config["opt"]["name"] == "sgd":
        optimizer_class = torch.optim.SGD

    # define models
    if args.model in ["swinunetr"]:

        patch_size = f"{config['preprocessing']['crop_pad_size'][0]}x" \
                        f"{config['preprocessing']['crop_pad_size'][1]}x" \
                        f"{config['preprocessing']['crop_pad_size'][2]}"

        # define model
        net = SwinUNETR(spatial_dims=config["model"]["swinunetr"]["spatial_dims"],
                        in_channels=1, out_channels=1,
                        img_size=config["preprocessing"]["crop_pad_size"],
                        depths=config["model"]["swinunetr"]["depths"],
                        feature_size=config["model"]["swinunetr"]["feature_size"],
                        num_heads=config["model"]["swinunetr"]["num_heads"],
                    )

        if config["model"]["swinunetr"]["use_pretrained"]:
            logger.info(f"Using SwinUNETR model with pre-trained weights ...")
            # download the pre-trained weights
            resource = (
                "https://github.com/Project-MONAI/MONAI-extra-test-data/releases/download/0.8.1/ssl_pretrained_weights.pth"
                )
            dst = os.path.join(config["directories"]["models_dir"], "swinunetr_pretrained")
            if not os.path.exists(dst):
                os.makedirs(dst, exist_ok=True)
            dst = os.path.join(dst, "ssl_pretrained_weights.pth")
            download_url(resource, dst)
            pretrained_path = os.path.normpath(dst)

            net = load_pretrained_swinunetr(net, path_pretrained_weights=pretrained_path)

        else:
            logger.info(f"Using SwinUNETR model initialized from scratch ...")

        save_exp_id = f"{args.model}_seed={config['seed']}_" \
                        f"ndata={n_datasets}_ncont={n_contrasts}_" \
                        f"ptr={int(config['model']['swinunetr']['use_pretrained'])}_" \
                        f"d={config['model']['swinunetr']['depths'][0]}_" \
                        f"nf={config['model']['swinunetr']['feature_size']}_" \
                        f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                        f"bs={config['opt']['batch_size']}" \
                        # f"bs={config['opt']['batch_size']}_{patch_size}" \
        # save_exp_id = f"_CSAdiceL_nspv={args.num_samples_per_volume}_bs={args.batch_size}_{img_size}" \

    elif args.model in ["nnunet-plain", "nnunet-resencM"]:

        # enabling deep supervision by default
        logger.info(f"Using {args.model} model WITH deep supervision ...")

        logger.info("Defining plans for nnUNet model ...")
        # =========================================================================================
        #                   Define plans json taken from nnUNet_preprocessed folder
        # =========================================================================================
        if args.model == "nnunet-plain":
            nnunet_plans = {
                "arch_class_name": "dynamic_network_architectures.architectures.unet.PlainConvUNet",
                "arch_kwargs": {
                    "n_stages": config["model"]["nnunet-plain"]["n_stages"],
                    "features_per_stage": config["model"]["nnunet-plain"]["features_per_stage"],
                    "strides": config["model"]["nnunet-plain"]["strides"],
                    "n_conv_per_stage": config["model"]["nnunet-plain"]["n_conv_per_stage"],
                    "n_conv_per_stage_decoder": config["model"]["nnunet-plain"]["n_conv_per_stage_decoder"]
                },
                "arch_kwargs_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            }
        elif args.model == "nnunet-resencM":
            nnunet_plans = {
                "arch_class_name": "dynamic_network_architectures.architectures.unet.ResidualEncoderUNet",
                "arch_kwargs": {
                    "n_stages": config["model"]["nnunet-resencM"]["n_stages"],
                    "features_per_stage": config["model"]["nnunet-resencM"]["features_per_stage"],
                    "strides": config["model"]["nnunet-resencM"]["strides"],
                    "n_blocks_per_stage": config["model"]["nnunet-resencM"]["n_blocks_per_stage"],
                    "n_conv_per_stage_decoder": config["model"]["nnunet-resencM"]["n_conv_per_stage_decoder"]
                },
                "arch_kwargs_requires_import": ["conv_op", "norm_op", "dropout_op", "nonlin"],
            }

        # define model
        net = create_nnunet_from_plans(plans=nnunet_plans, input_channels=1, output_channels=1, deep_supervision=True)
        # variable for saving patch size in the experiment id (same as crop_pad_size)
        patch_size = f"{config['preprocessing']['crop_pad_size'][0]}x" \
                        f"{config['preprocessing']['crop_pad_size'][1]}x" \
                        f"{config['preprocessing']['crop_pad_size'][2]}"
        # save experiment id
        save_exp_id = f"{args.model}_seed={config['seed']}_WithPraxNoSCT_" \
                        f"ndata={n_datasets}_ncont={n_contrasts}_" \
                        f"nf={config['model']['nnunet-plain']['features_per_stage'][-1]}_" \
                        f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                        f"bs={config['opt']['batch_size']}" \
                        
        # if pre-trained weights are provided, load them
        if args.pretrained_path is not None:
            logger.info(f"Loading pre-trained weights from {args.pretrained_path} ...")
            path_pretrained = os.path.join(args.pretrained_path, "best_model.ckpt")
            net = load_pretrained_weights(path_pretrained, net)
            save_exp_id = f"{save_exp_id}_ptrV21"
        else:
            logger.info(f"No input for pretrained weights. Starting training from scratch ...")

        if args.debug:
            save_exp_id = f"DEBUG_{save_exp_id}"

    # elif args.model in ["meunet"]:
    #     logger.info(f"Using MultiEncoderUNet model with {n_contrasts} contrasts ...")
    #     # define model
    #     net = MultiEncoderUNet(
    #         num_contrasts=n_contrasts, in_channels=1, 
    #         feature_maps=config["model"]["meunet"]["features_per_stage"],
    #         fusion_type=config["model"]["meunet"]["fusion_type"],
    #         norm_type=config["model"]["meunet"]["feat_norm_type"],
    #     )

    #     # save experiment id
    #     save_exp_id = f"{args.model}_seed={config['seed']}_" \
    #                     f"ndata={n_datasets}_ncont={n_contrasts}_" \
    #                     f"nf={config['model']['meunet']['features_per_stage'][-1]}_" \
    #                     f"ft={config['model']['meunet']['fusion_type']}_" \
    #                     f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
    #                     f"bs={config['opt']['batch_size']}" \
        
        if args.debug:
            save_exp_id = f"DEBUG_{save_exp_id}"

    timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")   # prints in YYYYMMDD-HHMMSS format
    save_exp_id = f"{timestamp}_{save_exp_id}"

    # save output to a log file
    logger.add(os.path.join(config["directories"]["models_dir"], f"{save_exp_id}", "logs.txt"), rotation="10 MB", level="INFO")

    # save config file to the output folder
    with open(os.path.join(config["directories"]["models_dir"], f"{save_exp_id}", "config.yaml"), "w") as f:
        yaml.dump(config, f)

    logger.info(f"Using {args.pad_mode} padding for the input images ...")
    logger.info(f"Using {args.input_label} labels as input to the model after data augmentation ...")

    # define loss function
    loss_func = AdapWingLoss(theta=0.5, omega=8, alpha=2.1, epsilon=1, reduction="sum")
    # NOTE: tried increasing omega and decreasing epsilon but results marginally worse than the above
    # loss_func = AdapWingLoss(theta=0.5, omega=12, alpha=2.1, epsilon=0.5, reduction="sum")
    logger.info(f"Using AdapWingLoss with theta={loss_func.theta}, omega={loss_func.omega}, alpha={loss_func.alpha}, epsilon={loss_func.epsilon} ...")

    # define callbacks
    early_stopping = pl.callbacks.EarlyStopping(
        monitor="val_loss", min_delta=0.00,
        patience=config["opt"]["early_stopping_patience"],
        verbose=False, mode="min")

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # training from scratch
    if not args.continue_from_checkpoint:
        # to save the best model on validation
        save_path = os.path.join(config["directories"]["models_dir"], f"{save_exp_id}")
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        # to save the results/model predictions
        results_path = os.path.join(config["directories"]["results_dir"], f"{save_exp_id}")
        if not os.path.exists(results_path):
            os.makedirs(results_path, exist_ok=True)

        # i.e. train by loading weights from scratch
        pl_model = Model(config, data_root=dataset_root,
                            optimizer_class=optimizer_class, loss_function=loss_func, net=net,
                            exp_id=save_exp_id, results_path=results_path)

        # # model compilation results in considerable speedup, so compiling the model
        # # more info: https://lightning.ai/blog/training-compiled-pytorch-2.0-with-pytorch-lightning/
        # logger.info(f"Compiling the model with torch.compile ...")
        # pl_model = torch.compile(pl_model) #, mode='reduce-overhead')

        # saving the best model based on validation loss
        logger.info(f"Saving best model to {save_path}!")
        checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, filename='best_model', monitor='val_loss',
            save_top_k=1, mode="min", save_last=True, save_weights_only=False)

        # # saving the best model based on soft validation dice score
        # checkpoint_callback_dice = pl.callbacks.ModelCheckpoint(
        #     dirpath=save_path, filename='best_model_dice', monitor='val_soft_dice',
        #     save_top_k=1, mode="max", save_last=False, save_weights_only=True)

        num_model_params = count_parameters(model=net)
        logger.info(f"Number of Trainable model parameters: {(num_model_params / 1e6):.3f}M")

        start_time = time.time()
        # wandb logger
        exp_logger = pl.loggers.WandbLogger(
                            name=save_exp_id,
                            save_dir="/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/saved_models",
                            group=config["dataset"]["name"],
                            log_model=True, # save best model using checkpoint callback
                            project='contrast-agnostic',
                            entity='naga-karthik',
                            config=config,
                            mode="disabled")

        # Saving training script to wandb
        wandb.save("main.py")

        # Enable TF32 on matmul and on cuDNN
        # torch._dynamo.config.verbose = True
        torch.backends.cuda.matmul.allow_tf32 = True    # same as setting torch.set_float32_matmul_precision("high"/"medium")
        torch.backends.cudnn.allow_tf32 = True

        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=1, accelerator="gpu",
            logger=exp_logger,
            callbacks=[checkpoint_callback_loss, lr_monitor, early_stopping],
            check_val_every_n_epoch=config["opt"]["check_val_every_n_epochs"],
            max_epochs=config["opt"]["max_epochs"],
            precision="bf16-mixed",
            # NOTE: Each epoch takes a looot of time with the aggregated dataset, so limiting the number of training batches
            # per epoch. Turns out that we don't need to go through all the training samples within an epoch for good performance.
            # nnunet hardcodes 250 training steps per epoch and we all know how it performs :)
            limit_train_batches=0.75,  # use 1.0 for full training
            enable_progress_bar=True)
            # profiler="simple",)     # to profile the training time taken for each step

        # Train!
        trainer.fit(pl_model)
        logger.info(f" Training Done!")
        end_time = time.time()

        duration = (end_time - start_time)
        logger.info(f"Total training time: {round((duration / 3600), 2)}hrs {(duration / 60) % 60}mins {(duration) % 60}secs")

    else:
        logger.info(f" Resuming training from the latest checkpoint! ")

        # check if wandb run folder is provided to resume using the same run
        if config["directories"]["wandb_run_folder"] is None:
            raise ValueError("Please provide the wandb run folder to resume training using the same run on WandB!")
        else:
            wandb_run_folder = os.path.basename(config["directories"]["wandb_run_folder"])
            wandb_run_id = wandb_run_folder.split("-")[-1]

        save_exp_id = config["directories"]["models_dir"]
        save_path = os.path.dirname(config["directories"]["models_dir"])
        logger.info(f"save_path: {save_path}")
        results_path = config["directories"]["results_dir"]

        # i.e. train by loading existing weights
        pl_model = Model(config, data_root=dataset_root,
                            optimizer_class=optimizer_class, loss_function=loss_func, net=net,
                            exp_id=save_exp_id, results_path=results_path)

        # saving the best model based on validation CSA loss
        checkpoint_callback_loss = pl.callbacks.ModelCheckpoint(
            dirpath=save_exp_id, filename='best_model', monitor='val_loss',
            save_top_k=1, mode="min", save_last=True, save_weights_only=True)

        # # saving the best model based on soft validation dice score
        # checkpoint_callback_dice = pl.callbacks.ModelCheckpoint(
        #     dirpath=save_exp_id, filename='best_model_dice', monitor='val_soft_dice',
        #     save_top_k=1, mode="max", save_last=False, save_weights_only=True)

        # wandb logger
        exp_logger = pl.loggers.WandbLogger(
                            save_dir=save_path,
                            group=config["dataset"]["name"],
                            log_model=True, # save best model using checkpoint callback
                            project='contrast-agnostic',
                            entity='naga-karthik',
                            config=args,
                            id=wandb_run_id, resume='must')

        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=1, accelerator="gpu",
            logger=exp_logger,
            callbacks=[checkpoint_callback_loss, lr_monitor, early_stopping],
            check_val_every_n_epoch=config["opt"]["check_val_every_n_epochs"],
            max_epochs=config["opt"]["max_epochs"],
            precision="bf16-mixed",
            enable_progress_bar=True)
            # profiler="simple",)     # to profile the training time taken for each step

        # Train!
        trainer.fit(pl_model, ckpt_path=os.path.join(save_exp_id, "last.ckpt"),)
        logger.info(f" Training Done!")

    # Test!
    trainer.test(pl_model)
    logger.info(f"TESTING DONE!")

    # closing the current wandb instance so that a new one is created for the next fold
    wandb.finish()

    # TODO: Figure out saving test metrics to a file
    with open(os.path.join(results_path, 'test_metrics.txt'), 'a') as f:
        print('\n-------------- Test Metrics ----------------', file=f)
        print(f"{args.model}_seed={config['seed']}_" \
                        f"ncontrasts={n_contrasts}_" \
                        f"nf={config['model']['nnunet']['base_num_features']}_" \
                        f"opt={config['opt']['name']}_lr={config['opt']['lr']}_AdapW_" \
                        f"bs={config['opt']['batch_size']}_{patch_size}" \
                        f"_{timestamp}", file=f)

        print('\n-------------- Test Hard Dice Scores ----------------', file=f)
        print("Hard Dice --> Mean: %0.3f, Std: %0.3f" % (pl_model.avg_test_dice_hard, pl_model.std_test_dice_hard), file=f)

        print('\n-------------- Test Soft Dice Scores ----------------', file=f)
        print("Soft Dice --> Mean: %0.3f, Std: %0.3f" % (pl_model.avg_test_dice, pl_model.std_test_dice), file=f)

        print('-------------------------------------------------------', file=f)


if __name__ == "__main__":
    args = get_args()
    main(args)