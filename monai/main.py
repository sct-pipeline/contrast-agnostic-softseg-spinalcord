import os
import argparse
from datetime import datetime
from loguru import logger

import numpy as np
import wandb
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils import precision_score, recall_score, dice_score, compute_average_csa, PolyLRScheduler
from losses import SoftDiceLoss, AdapWingLoss
from transforms import train_transforms, val_transforms
from models import ModifiedUNet3D

from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, UNETR
from monai.data import (DataLoader, Dataset, CacheDataset, load_decathlon_datalist, decollate_batch)
from monai.transforms import (Compose, EnsureType, EnsureTyped, Invertd, SaveImaged, SaveImage)

# create a "model"-agnostic class with PL to use different models
class Model(pl.LightningModule):
    def __init__(self, args, data_root, fold_num, net, loss_function, optimizer_class, 
                 exp_id=None, results_path=None):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['net'])

        self.root = data_root
        self.fold_num = fold_num
        self.net = net
        # self.load_pretrained = load_pretrained
        self.lr = args.learning_rate
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class
        self.save_exp_id = exp_id
        self.results_path = results_path

        self.best_val_dice, self.best_val_epoch = 0, 0
        self.best_val_csa = float("inf")

        # define cropping and padding dimensions
        self.voxel_cropping_size = (160, 224, 96)   # (80, 192, 160) taken from nnUNet_plans.json
        self.inference_roi_size = (160, 224, 96)
        self.spacing = (1.0, 1.0, 1.0)

        # define post-processing transforms for validation, nothing fancy just making sure that it's a tensor (default)
        self.val_post_pred = Compose([EnsureType()]) 
        self.val_post_label = Compose([EnsureType()])

        # define evaluation metric
        self.soft_dice_metric = dice_score

        # temp lists for storing outputs from training, validation, and testing
        self.train_step_outputs = []
        self.val_step_outputs = []
        self.test_step_outputs = []

        # MSE loss for comparing the CSA values
        self.mse_loss = torch.nn.MSELoss()


    # --------------------------------
    # FORWARD PASS
    # --------------------------------
    def forward(self, x):
        # x, context_features = self.encoder(x)
        # preds = self.decoder(x, context_features)
        
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
        set_determinism(seed=self.args.seed)
        
        # define training and validation transforms
        transforms_train = train_transforms(
            crop_size=self.voxel_cropping_size, 
            num_samples_pv=self.args.num_samples_per_volume,
            lbl_key='label'
        )
        transforms_val = val_transforms(lbl_key='label')
        
        # load the dataset
        dataset = os.path.join(self.root, f"spine-generic-ivado-comparison_dataset.json")
        train_files = load_decathlon_datalist(dataset, True, "train")
        val_files = load_decathlon_datalist(dataset, True, "validation")
        test_files = load_decathlon_datalist(dataset, True, "test")

        if args.debug:
            train_files = train_files[:10]
            val_files = val_files[:10]
            test_files = test_files[:6]
        
        train_cache_rate = 0.25 if args.debug else 0.5
        self.train_ds = CacheDataset(data=train_files, transform=transforms_train, cache_rate=train_cache_rate, num_workers=4)
        self.val_ds = CacheDataset(data=val_files, transform=transforms_val, cache_rate=0.25, num_workers=4)

        # define test transforms
        transforms_test = val_transforms(lbl_key='label')
        
        # define post-processing transforms for testing; taken (with explanations) from 
        # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
        self.test_post_pred = Compose([
            EnsureTyped(keys=["pred", "label"]),
            Invertd(keys=["pred", "label"], transform=transforms_test, 
                    orig_keys=["image", "label"], 
                    meta_keys=["pred_meta_dict", "label_meta_dict"],
                    nearest_interp=False, to_tensor=True),
            ])
        self.test_ds = CacheDataset(data=test_files, transform=transforms_test, cache_rate=0.1, num_workers=4)


    # --------------------------------
    # DATA LOADERS
    # --------------------------------
    def train_dataloader(self):
        # NOTE: if num_samples=4 in RandCropByPosNegLabeld and batch_size=2, then 2 x 4 images are generated for network training
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, 
                            pin_memory=True,) # collate_fn=pad_list_data_collate)
        # list_data_collate is only useful when each input in the batch has different shape

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)

    
    # --------------------------------
    # OPTIMIZATION
    # --------------------------------
    def configure_optimizers(self):
        if self.args.optimizer == "sgd":
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, momentum=0.99, weight_decay=1e-5, nesterov=True)
        else:
            optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
        scheduler = PolyLRScheduler(optimizer, self.lr, max_steps=self.args.max_epochs)
        return [optimizer], [scheduler]


    # --------------------------------
    # TRAINING
    # --------------------------------
    def training_step(self, batch, batch_idx):

        inputs, labels = batch["image"], batch["label"]

        # filter empty label patches
        if not labels.any():
            print("Encountered empty label patch. Skipping...")
            return None

        output = self.forward(inputs)   # logits

        # calculate training loss   
        # NOTE: the diceLoss expects the input to be logits (which it then normalizes inside)
        dice_loss = self.loss_function(output, labels)

        # get probabilities from logits
        output = F.relu(output) / F.relu(output).max() if bool(F.relu(output).max()) else F.relu(output)

        # calculate train dice
        # NOTE: this is done on patches (and not entire 3D volume) because SlidingWindowInference is not used here
        # So, take this dice score with a lot of salt
        train_soft_dice = self.soft_dice_metric(output, labels) 
        # train_hard_dice = self.soft_dice_metric((output.detach() > 0.5).float(), (labels.detach() > 0.5).float())

        # binarize the predictions and the labels
        output = (output.detach() > 0.5).float()
        labels = (labels.detach() > 0.5).float()
        
        # compute CSA for each element of the batch
        csa_loss = 0.0
        for batch_idx in range(output.shape[0]):
            pred_patch_csa = compute_average_csa(output[batch_idx].squeeze(), self.spacing)
            gt_patch_csa = compute_average_csa(labels[batch_idx].squeeze(), self.spacing)
            csa_loss += (pred_patch_csa - gt_patch_csa) ** 2

        # average CSA loss across the batch
        csa_loss = csa_loss / output.shape[0]

        # total loss
        loss = dice_loss + csa_loss

        metrics_dict = {
            "loss": loss.cpu(),
            "dice_loss": dice_loss.cpu(),
            "csa_loss": csa_loss.cpu(),
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
            train_loss, train_dice_loss, train_csa_loss = 0, 0, 0
            num_items, train_soft_dice = 0, 0
            for output in self.train_step_outputs:
                train_loss += output["loss"].item()
                train_dice_loss += output["dice_loss"].item()
                train_csa_loss += output["csa_loss"].item()
                train_soft_dice += output["train_soft_dice"].item()
                num_items += output["train_number"]
            
            mean_train_loss = (train_loss / num_items)
            mean_train_dice_loss = (train_dice_loss / num_items)
            mean_train_csa_loss = (train_csa_loss / num_items)
            mean_train_soft_dice = (train_soft_dice / num_items)

            wandb_logs = {
                "train_soft_dice": mean_train_soft_dice, 
                "train_loss": mean_train_loss,
                "train_dice_loss": mean_train_dice_loss,
                "train_csa_loss": mean_train_csa_loss,
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

        outputs = sliding_window_inference(inputs, self.inference_roi_size, 
                                           sw_batch_size=4, predictor=self.forward, overlap=0.5,) 
        # outputs shape: (B, C, <original H x W x D>)
        
        # calculate validation loss
        dice_loss = self.loss_function(outputs, labels)

        # get probabilities from logits
        outputs = F.relu(outputs) / F.relu(outputs).max() if bool(F.relu(outputs).max()) else F.relu(outputs)
        
        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]
        val_soft_dice = self.soft_dice_metric(post_outputs[0], post_labels[0])

        hard_preds, hard_labels = (post_outputs[0].detach() > 0.5).float(), (post_labels[0].detach() > 0.5).float()
        val_hard_dice = self.soft_dice_metric(hard_preds, hard_labels)

        # compute val CSA loss
        val_csa_loss = 0.0
        for batch_idx in range(hard_preds.shape[0]):
            pred_patch_csa = compute_average_csa(hard_preds[batch_idx].squeeze(), self.spacing)
            gt_patch_csa = compute_average_csa(hard_labels[batch_idx].squeeze(), self.spacing)
            val_csa_loss += (pred_patch_csa - gt_patch_csa) ** 2

        # average CSA loss across the batch
        val_csa_loss = val_csa_loss / hard_preds.shape[0]

        # total loss
        loss = dice_loss + val_csa_loss

        # NOTE: there was a massive memory leak when storing cuda tensors in this dict. Hence,
        # using .detach() to avoid storing the whole computation graph
        # Ref: https://discuss.pytorch.org/t/cuda-memory-leak-while-training/82855/2
        metrics_dict = {
            "val_loss": loss.detach().cpu(),
            "val_dice_loss": dice_loss.detach().cpu(),
            "val_csa_loss": val_csa_loss.detach().cpu(),
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
        val_dice_loss, val_csa_loss = 0, 0
        for output in self.val_step_outputs:
            val_loss += output["val_loss"].sum().item()
            val_soft_dice += output["val_soft_dice"].sum().item()
            val_hard_dice += output["val_hard_dice"].sum().item()
            val_dice_loss += output["val_dice_loss"].sum().item()
            val_csa_loss += output["val_csa_loss"].sum().item()
            num_items += output["val_number"]
        
        mean_val_loss = (val_loss / num_items)
        mean_val_soft_dice = (val_soft_dice / num_items)
        mean_val_hard_dice = (val_hard_dice / num_items)
        mean_val_dice_loss = (val_dice_loss / num_items)
        mean_val_csa_loss = (val_csa_loss / num_items)
                
        wandb_logs = {
            "val_soft_dice": mean_val_soft_dice,
            "val_hard_dice": mean_val_hard_dice,
            "val_loss": mean_val_loss,
            "val_dice_loss": mean_val_dice_loss,
            "val_csa_loss": mean_val_csa_loss,
        }
        # # save the best model based on validation dice score
        # if mean_val_soft_dice > self.best_val_dice:
        #     self.best_val_dice = mean_val_soft_dice
        #     self.best_val_epoch = self.current_epoch
        
        # save the best model based on validation CSA loss
        if mean_val_csa_loss < self.best_val_csa:
            self.best_val_csa = mean_val_csa_loss
            self.best_val_epoch = self.current_epoch

        print(
            f"Current epoch: {self.current_epoch}"
            f"\nAverage Soft Dice (VAL): {mean_val_soft_dice:.4f}"
            f"\nAverage Hard Dice (VAL): {mean_val_hard_dice:.4f}"
            f"\nAverage CSA (VAL): {mean_val_csa_loss:.4f}"
            # f"\nBest Average Soft Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_epoch}"
            f"\nBest Average CSA: {self.best_val_csa:.4f} at Epoch: {self.best_val_epoch}"
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
        
        test_input, test_label = batch["image"], batch["label"]
        # print(batch["label_meta_dict"]["filename_or_obj"][0])
        # print(f"test_input.shape: {test_input.shape} \t test_label.shape: {test_label.shape}")
        batch["pred"] = sliding_window_inference(test_input, self.inference_roi_size, 
                                                 sw_batch_size=4, predictor=self.forward, overlap=0.5)
        # print(f"batch['pred'].shape: {batch['pred'].shape}")
        
        # normalize the logits
        batch["pred"] = F.relu(batch["pred"]) / F.relu(batch["pred"]).max() if bool(F.relu(batch["pred"]).max()) else F.relu(batch["pred"])

        # # upon fsleyes visualization, observed that very small values need to be set to zero, but NOT fully binarizing the pred
        # batch["pred"][batch["pred"] < 0.099] = 0.0

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]

        # make sure that the shapes of prediction and GT label are the same
        # print(f"pred shape: {post_test_out[0]['pred'].shape}, label shape: {post_test_out[0]['label'].shape}")
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label'].shape
        
        pred, label = post_test_out[0]['pred'].cpu(), post_test_out[0]['label'].cpu()

        # save the prediction and label
        if self.args.save_test_preds:

            subject_name = (batch["image_meta_dict"]["filename_or_obj"][0]).split("/")[-1].replace(".nii.gz", "")
            print(f"Saving subject: {subject_name}")

            # image saver class
            save_folder = os.path.join(self.results_path, subject_name.split("_")[0])
            pred_saver = SaveImage(
                output_dir=save_folder, output_postfix="pred", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the prediction
            pred_saver(pred)

            label_saver = SaveImage(
                output_dir=save_folder, output_postfix="gt", output_ext=".nii.gz", 
                separate_folder=False, print_log=False)
            # save the label
            label_saver(label)
            

        # NOTE: Important point from the SoftSeg paper - binarize predictions before computing metrics
        # calculate all metrics here
        # 1. Dice Score
        test_soft_dice = self.soft_dice_metric(pred, label)

        # binarizing the predictions 
        pred = (post_test_out[0]['pred'].detach().cpu() > 0.5).float()
        label = (post_test_out[0]['label'].detach().cpu() > 0.5).float()

        # 1.1 Hard Dice Score
        test_hard_dice = self.soft_dice_metric(pred.numpy(), label.numpy())
        # 2. Precision Score
        test_precision = precision_score(pred.numpy(), label.numpy())
        # 3. Recall Score
        test_recall = recall_score(pred.numpy(), label.numpy())

        metrics_dict = {
            "test_hard_dice": test_hard_dice,
            "test_soft_dice": test_soft_dice,
            "test_precision": test_precision,
            "test_recall": test_recall,
        }
        self.test_step_outputs.append(metrics_dict)

        return metrics_dict

    def on_test_epoch_end(self):
        
        avg_hard_dice_test, std_hard_dice_test = np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_hard_dice"] for x in self.test_step_outputs]).std()
        avg_soft_dice_test, std_soft_dice_test = np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).mean(), \
                                                    np.stack([x["test_soft_dice"] for x in self.test_step_outputs]).std()
        avg_precision_test = np.stack([x["test_precision"] for x in self.test_step_outputs]).mean()
        avg_recall_test = np.stack([x["test_recall"] for x in self.test_step_outputs]).mean()
        
        logger.info(f"Test (Soft) Dice: {avg_soft_dice_test}")
        logger.info(f"Test (Hard) Dice: {avg_hard_dice_test}")
        logger.info(f"Test Precision Score: {avg_precision_test}")
        logger.info(f"Test Recall Score: {avg_recall_test}")
        
        self.avg_test_dice, self.std_test_dice = avg_soft_dice_test, std_soft_dice_test
        self.avg_test_dice_hard, self.std_test_dice_hard = avg_hard_dice_test, std_hard_dice_test
        self.avg_test_precision = avg_precision_test
        self.avg_test_recall = avg_recall_test

        # free up memory
        self.test_step_outputs.clear()


# --------------------------------
# MAIN
# --------------------------------
def main(args):
    # Setting the seed
    pl.seed_everything(args.seed, workers=True)

    # define root path for finding datalists
    dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/contrast-agnostic-softseg-spinalcord/monai"

    # define optimizer
    if args.optimizer in ["adam", "Adam"]:
        optimizer_class = torch.optim.Adam
    elif args.optimizer in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    # define models
    if args.model in ["unet", "UNet"]:            
        # # this is the MONAI model
        # net = UNet(spatial_dims=3, 
        #             in_channels=1, out_channels=1,
        #             channels=(
        #                 args.init_filters, 
        #                 args.init_filters * 2, 
        #                 args.init_filters * 4, 
        #                 args.init_filters * 8, 
        #                 args.init_filters * 16
        #             ),
        #             strides=(2, 2, 2, 2),
        #             num_res_units=4,
        #         )
        # patch_size = "160x224x96"
        # save_exp_id =f"{args.model}_nf={args.init_filters}_nrs=4_opt={args.optimizer}_lr={args.learning_rate}" \
        #                 f"_diceL_nspv={args.num_samples_per_volume}_bs={args.batch_size}_{patch_size}"
        
        # This is the ivadomed model
        net = ModifiedUNet3D(in_channels=1, out_channels=1, init_filters=args.init_filters)
        patch_size =  "160x224x96"   # "64x128x64"
        save_exp_id =f"ivado_{args.model}_nf={args.init_filters}_opt={args.optimizer}_lr={args.learning_rate}" \
                        f"_CSAdiceL_bestValCSA_nspv={args.num_samples_per_volume}" \
                        f"_bs={args.batch_size}_{patch_size}"

    elif args.model in ["unetr", "UNETR"]:
        # define image size to be fed to the model
        img_size = (96, 96, 96)
        
        # define model
        net = UNETR(spatial_dims=3,
                    in_channels=1, out_channels=1, 
                    img_size=img_size,
                    feature_size=args.feature_size, 
                    hidden_size=args.hidden_size, 
                    mlp_dim=args.mlp_dim, 
                    num_heads=args.num_heads,
                    pos_embed="perceptron", 
                    norm_name="instance", 
                    res_block=True, 
                    dropout_rate=0.2,
                )
        save_exp_id = f"{args.model}_lr={args.learning_rate}" \
                        f"_fs={args.feature_size}_hs={args.hidden_size}_mlpd={args.mlp_dim}_nh={args.num_heads}"

    # define loss function
    loss_func = SoftDiceLoss(p=1, smooth=1.0)
    # loss_func = DiceCrossEntropyLoss(weight_ce=1.0, weight_dice=1.0)
    # loss_func = AdapWingLoss(epsilon=1, theta=0.5, alpha=2.1, omega=8.0, reduction='mean')

    # TODO: move this inside the for loop when using more folds
    timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")   # prints in YYYYMMDD-HHMMSS format
    save_exp_id = f"{save_exp_id}_{timestamp}"

    # to save the best model on validation
    save_path = os.path.join(args.save_path, f"{save_exp_id}")
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    # to save the results/model predictions 
    results_path = os.path.join(args.results_dir, f"{save_exp_id}")
    if not os.path.exists(results_path):
        os.makedirs(results_path, exist_ok=True)

    # train across all folds of the dataset
    for fold in range(args.num_cv_folds):
        logger.info(f" Training on fold {fold+1} out of {args.num_cv_folds} folds! ")

        # timestamp = datetime.now().strftime(f"%Y%m%d-%H%M")   # prints in YYYYMMDD-HHMMSS format
        # save_exp_id = f"{save_exp_id}_fold={fold}_{timestamp}"

        # i.e. train by loading weights from scratch
        pl_model = Model(args, data_root=dataset_root, fold_num=fold, 
                         optimizer_class=optimizer_class, loss_function=loss_func, net=net, 
                         exp_id=save_exp_id, results_path=results_path)

        # don't use wandb logger if in debug mode
        # if not args.debug:
        exp_logger = pl.loggers.WandbLogger(
                            name=save_exp_id,
                            save_dir=args.save_path,
                            group=f"{args.model}_Adam", 
                            log_model=True, # save best model using checkpoint callback
                            project='contrast-agnostic',
                            entity='naga-karthik',
                            config=args)
        
        # # saving the best model based on soft validation dice score
        # checkpoint_callback = pl.callbacks.ModelCheckpoint(
        #     dirpath=save_path, filename='best_model', monitor='val_soft_dice', 
        #     save_top_k=5, mode="max", save_last=False, save_weights_only=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=save_path, filename='best_model', monitor='val_csa_loss', 
            save_top_k=5, mode="min", save_last=False, save_weights_only=True)
        
        # early_stopping = pl.callbacks.EarlyStopping(monitor="val_soft_dice", min_delta=0.00, patience=args.patience, 
        #                     verbose=False, mode="max")
        early_stopping = pl.callbacks.EarlyStopping(monitor="val_csa_loss", min_delta=0.00, patience=args.patience, 
                            verbose=False, mode="min")

        lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

        # initialise Lightning's trainer.
        trainer = pl.Trainer(
            devices=1, accelerator="gpu", # strategy="ddp",
            logger=exp_logger,
            callbacks=[checkpoint_callback, lr_monitor, early_stopping],
            check_val_every_n_epoch=args.check_val_every_n_epochs,
            max_epochs=args.max_epochs, 
            precision=32,   # TODO: see if 16-bit precision is stable
            # deterministic=True,
            enable_progress_bar=args.enable_progress_bar)

        # Train!
        trainer.fit(pl_model)        
        logger.info(f" Training Done!")

        # Saving training script to wandb
        wandb.save("main.py")

        # Test!
        trainer.test(pl_model)
        logger.info(f"TESTING DONE!")

        # closing the current wandb instance so that a new one is created for the next fold
        wandb.finish()
        
        # TODO: Figure out saving test metrics to a file
        with open(os.path.join(results_path, 'test_metrics.txt'), 'a') as f:
            print('\n-------------- Test Metrics ----------------', file=f)
            print(f"\nSeed Used: {args.seed}", file=f)
            print(f"\ninitf={args.init_filters}_lr={args.learning_rate}_bs={args.batch_size}_{timestamp}", file=f)
            print(f"\npatch_size={pl_model.voxel_cropping_size}", file=f)
            
            print('\n-------------- Test Hard Dice Scores ----------------', file=f)
            print("Hard Dice --> Mean: %0.3f, Std: %0.3f" % (pl_model.avg_test_dice_hard, pl_model.std_test_dice_hard), file=f)

            print('\n-------------- Test Soft Dice Scores ----------------', file=f)
            print("Soft Dice --> Mean: %0.3f, Std: %0.3f" % (pl_model.avg_test_dice, pl_model.std_test_dice), file=f)

            print('\n-------------- Test Precision Scores ----------------', file=f)
            print("Precision --> Mean: %0.3f" % (pl_model.avg_test_precision), file=f)

            print('\n-------------- Test Recall Scores -------------------', file=f)
            print("Recall --> Mean: %0.3f" % (pl_model.avg_test_recall), file=f)

            print('-------------------------------------------------------', file=f)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for training custom models for SCI Lesion Segmentation.')
    # Arguments for model, data, and training and saving
    parser.add_argument('-m', '--model', 
                        choices=['unet', 'UNet', 'unetr', 'UNETR', 'attentionunet'], 
                        default='unet', type=str, help='Model type to be used')
    # dataset
    parser.add_argument('-nspv', '--num_samples_per_volume', default=4, type=int, help="Number of samples to crop per volume")    
    parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="Number of cross validation folds")
    
    # unet model 
    parser.add_argument('-initf', '--init_filters', default=16, type=int, help="Number of Filters in Init Layer")
    # parser.add_argument('-ps', '--patch_size', type=int, default=128, help='List containing subvolume size')
    parser.add_argument('-dep', '--unet_depth', default=3, type=int, help="Depth of UNet model")

    # unetr model 
    parser.add_argument('-fs', '--feature_size', default=16, type=int, help="Feature Size")
    parser.add_argument('-hs', '--hidden_size', default=768, type=int, help='Dimensionality of hidden embeddings')
    parser.add_argument('-mlpd', '--mlp_dim', default=2048, type=int, help='Dimensionality of MLP layer')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads in Multi-head Attention')

    # optimizations
    parser.add_argument('-me', '--max_epochs', default=1000, type=int, help='Number of epochs for the training process')
    parser.add_argument('-bs', '--batch_size', default=2, type=int, help='Batch size of the training and validation processes')
    parser.add_argument('-opt', '--optimizer', 
                        choices=['adam', 'Adam', 'SGD', 'sgd'], 
                        default='adam', type=str, help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate for training the model')
    parser.add_argument('-pat', '--patience', default=25, type=int, 
                            help='number of validation steps (val_every_n_iters) to wait before early stopping')
    # NOTE: patience is acutally until (patience * check_val_every_n_epochs) epochs 
    parser.add_argument('-epb', '--enable_progress_bar', default=False, action='store_true', 
                            help='by default is disabled since it doesnt work in colab')
    parser.add_argument('-cve', '--check_val_every_n_epochs', default=1, type=int, help='num of epochs to wait before validation')
    # saving
    parser.add_argument('-sp', '--save_path', 
                        default=f"/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/saved_models", 
                        type=str, help='Path to the saved models directory')
    parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', 
                            help='Load model from checkpoint and continue training')
    parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
    parser.add_argument('-debug', default=False, action='store_true', help='if true, results are not logged to wandb')
    parser.add_argument('-stp', '--save_test_preds', default=False, action='store_true',
                            help='if true, test predictions are saved in `save_path`')
    # testing
    parser.add_argument('-rd', '--results_dir', 
                    default=f"/home/GRAMES.POLYMTL.CA/u114716/contrast-agnostic/results", 
                    type=str, help='Path to the model prediction results directory')


    args = parser.parse_args()

    main(args)