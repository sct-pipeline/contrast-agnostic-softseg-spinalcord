import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy
import numpy as np


# TODO: also check out nnUNet's implementation of soft-dice loss (if required)
# https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/dice.py

class SoftDiceLoss(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    taken from: https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
    '''
    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, logits, labels):
        '''
        inputs:
            preds: logits - tensor of shape (N, H, W, ...)
            labels: soft labels [0,1] - tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        preds = F.relu(logits) / F.relu(logits).max() if bool(F.relu(logits).max()) else F.relu(logits)
        
        numer = (preds * labels).sum()
        denor = (preds.pow(self.p) + labels.pow(self.p)).sum()
        # loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        loss = - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(DiceCrossEntropyLoss, self).__init__()
        self.ce_weight = weight_ce
        self.dice_weight = weight_dice

        self.dice_loss = SoftDiceLoss()
        # self.ce_loss = RobustCrossEntropyLoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, preds, labels):
        '''
        inputs:
            preds: logits (not probabilities!) - tensor of shape (N, H, W, ...)
            labels: soft labels [0,1] - tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        ce_loss = self.ce_loss(preds, labels)

        # dice loss will convert logits to probabilities
        dice_loss = self.dice_loss(preds, labels)

        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss
        return loss
    

class AdapWingLoss(nn.Module):
    """
    Adaptive Wing loss used for heatmap regression
    Adapted from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/losses.py#L341

    .. seealso::
        Wang, Xinyao, Liefeng Bo, and Li Fuxin. "Adaptive wing loss for robust face alignment via heatmap regression."
        Proceedings of the IEEE International Conference on Computer Vision. 2019.

    Args:
        theta (float): Threshold to switch between the linear and non-linear parts of the piece-wise loss function.
        alpha (float): Used to adapt the behaviour of the loss function at y=0 and y=1 and make loss smooth at 0 (background).
        It needs to be slightly above 2 to maintain ideal properties.
        omega (float): Multiplicative factor for non linear part of the loss.
        epsilon (float): factor to avoid gradient explosion. It must not be too small
        NOTE: Larger omega and smaller epsilon values will increase the influence on small errors and vice versa
    """

    def __init__(self, theta=0.5, alpha=2.1, omega=14, epsilon=1, reduction='sum'):
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.epsilon = epsilon
        self.reduction = reduction
        super(AdapWingLoss, self).__init__()

    def forward(self, input, target):
        eps = self.epsilon
        batch_size = target.size()[0]

        # Adaptive Wing loss. Section 4.2 of the paper.
        # Compute adaptive factor
        A = self.omega * (1 / (1 + torch.pow(self.theta / eps,
                                             self.alpha - target))) * \
            (self.alpha - target) * torch.pow(self.theta / eps,
                                              self.alpha - target - 1) * (1 / eps)

        # Constant term to link linear and non linear part
        C = (self.theta * A - self.omega * torch.log(1 + torch.pow(self.theta / eps, self.alpha - target)))

        diff_hm = torch.abs(target - input)
        AWingLoss = A * diff_hm - C
        idx = diff_hm < self.theta
        # NOTE: this is a memory-efficient version than the one in ivadomed losses.py
        # where idx is True, compute the non-linear part of the loss, otherwise keep the linear part
        # the non-linear parts ensures small errors (as given by idx) have a larger influence to refine the predictions at the boundaries
        # the linear part makes the loss function behave more like the MSE loss, which has a linear influence 
        # (i.e. small errors where y=0 --> small influence --> small gradients)
        AWingLoss = torch.where(idx, self.omega * torch.log(1 + torch.pow(diff_hm / eps, self.alpha - target)), AWingLoss)


        # Mask for weighting the loss function. Section 4.3 of the paper.
        mask = torch.zeros_like(target)
        kernel = scipy.ndimage.generate_binary_structure(2, 2)
        # For 3D segmentation tasks
        if len(input.shape) == 5:
            kernel = scipy.ndimage.generate_binary_structure(3, 2)

        for i in range(batch_size):
            img_list = list()
            img_list.append(np.round(target[i].cpu().numpy() * 255))
            img_merge = np.concatenate(img_list)
            img_dilate = scipy.ndimage.binary_opening(img_merge, np.expand_dims(kernel, axis=0))
            # NOTE: why 51? the paper thresholds the dilated GT heatmap at 0.2. So, 51/255 = 0.2 
            img_dilate[img_dilate < 51] = 1  # 0*omega+1
            img_dilate[img_dilate >= 51] = 1 + self.omega  # 1*omega+1
            img_dilate = np.array(img_dilate, dtype=int)

            mask[i] = torch.tensor(img_dilate)

        AWingLoss *= mask

        sum_loss = torch.sum(AWingLoss)
        if self.reduction == "sum":
            return sum_loss
        elif self.reduction == "mean":
            all_pixel = torch.sum(mask)
            return sum_loss / all_pixel
