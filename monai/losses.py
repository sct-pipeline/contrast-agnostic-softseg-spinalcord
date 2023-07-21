import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F


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


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension
    input must be logits, not probabilities!
    adapted from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/loss/robust_ce_loss.py
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:

        # binarize soft labels with threshold 0.5 before using cross-entropy
        target = torch.where(target > 0.5, torch.ones_like(target), torch.zeros_like(target))

        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]

        return super().forward(input, target.long())


class DiceCrossEntropyLoss(nn.Module):
    def __init__(self, weight_ce=1.0, weight_dice=1.0):
        super(DiceCrossEntropyLoss).__init__()
        self.ce_weight = weight_ce
        self.dice_weight = weight_dice

        self.dice_loss = SoftDiceLoss()
        self.ce_loss = RobustCrossEntropyLoss()

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