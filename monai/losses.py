import torch
import torch.nn as nn


class SoftDiceLoss(nn.Module):
    '''
    soft-dice loss, useful in binary segmentation
    taken from: https://github.com/CoinCheung/pytorch-loss/blob/master/soft_dice_loss.py
    '''
    def __init__(self, p=1, smooth=1):
        super(SoftDiceLoss, self).__init__()
        self.p = p
        self.smooth = smooth

    def forward(self, preds, labels):
        '''
        inputs:
            preds: normalized probabilities (not logits) - tensor of shape (N, H, W, ...)
            labels: soft labels [0,1] - tensor of shape(N, H, W, ...)
        output:
            loss: tensor of shape(1, )
        '''
        # probs = torch.sigmoid(logits)
        numer = (preds * labels).sum()
        denor = (preds.pow(self.p) + labels.pow(self.p)).sum()
        loss = 1. - (2 * numer + self.smooth) / (denor + self.smooth)
        return loss