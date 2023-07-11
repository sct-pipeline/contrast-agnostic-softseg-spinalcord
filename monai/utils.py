import numpy as np
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import _LRScheduler


class FoldGenerator:
    """
    Adapted from https://github.com/MIC-DKFZ/medicaldetectiontoolkit/blob/master/utils/dataloader_utils.py#L59 
    Generates splits of indices for a given length of a dataset to perform n-fold cross-validation.
    splits each fold into 3 subsets for training, validation and testing.
    This form of cross validation uses an inner loop test set, which is useful if test scores shall be reported on a
    statistically reliable amount of patients, despite limited size of a dataset.
    If hold out test set is provided and hence no inner loop test set needed, just add test_idxs to the training data in the dataloader.
    This creates straight-forward train-val splits.
    :returns names list: list of len n_splits. each element is a list of len 3 for train_ix, val_ix, test_ix.
    """
    def __init__(self, seed, n_splits, len_data):
        """
        :param seed: Random seed for splits.
        :param n_splits: number of splits, e.g. 5 splits for 5-fold cross-validation
        :param len_data: number of elements in the dataset.
        """
        self.tr_ix = []
        self.val_ix = []
        self.te_ix = []
        self.slicer = None
        self.missing = 0
        self.fold = 0
        self.len_data = len_data
        self.n_splits = n_splits
        self.myseed = seed
        self.boost_val = 0

    def init_indices(self):

        t = list(np.arange(self.len_cv_names))
        # round up to next splittable data amount.
        if self.n_splits == 5:
            split_length = int(np.ceil(len(t) / float(self.n_splits)) // 1.5)
        else:
            split_length = int(np.ceil(len(t) / float(self.n_splits)))
        self.slicer = split_length
        print(self.slicer)
        self.mod = len(t) % self.n_splits
        if self.mod > 0:
            # missing is the number of folds, in which the new splits are reduced to account for missing data.
            self.missing = self.n_splits - self.mod

        # for 100 subjects, performs a 60-20-20 split with n_splits
        self.te_ix = t[:self.slicer]
        self.tr_ix = t[self.slicer:]
        self.val_ix = self.tr_ix[:self.slicer]
        self.tr_ix = self.tr_ix[self.slicer:]

    def new_fold(self):

        slicer = self.slicer
        if self.fold < self.missing:
            slicer = self.slicer - 1

        temp = self.te_ix

        # catch exception mod == 1: test set collects 1+ data since walk through both roudned up splits.
        # account for by reducing last fold split by 1.
        if self.fold == self.n_splits-2 and self.mod ==1:
            temp += self.val_ix[-1:]
            self.val_ix = self.val_ix[:-1]

        self.te_ix = self.val_ix
        self.val_ix = self.tr_ix[:slicer]
        self.tr_ix = self.tr_ix[slicer:] + temp


    def get_fold_names(self):
        names_list = []
        rgen = np.random.RandomState(self.myseed)
        cv_names = np.arange(self.len_data)

        rgen.shuffle(cv_names)
        self.len_cv_names = len(cv_names)
        self.init_indices()

        for split in range(self.n_splits):
            train_names, val_names, test_names = cv_names[self.tr_ix], cv_names[self.val_ix], cv_names[self.te_ix]
            names_list.append([train_names, val_names, test_names, self.fold])
            self.new_fold()
            self.fold += 1

        return names_list


def numeric_score(prediction, groundtruth):
    """Computation of statistical numerical scores:

    * FP = Soft False Positives
    * FN = Soft False Negatives
    * TP = Soft True Positives
    * TN = Soft True Negatives

    Robust to hard or soft input masks. For example::
        prediction=np.asarray([0, 0.5, 1])
        groundtruth=np.asarray([0, 1, 1])
        Leads to FP = 1.5

    Note: It assumes input values are between 0 and 1.

    Args:
        prediction (ndarray): Binary prediction.
        groundtruth (ndarray): Binary groundtruth.

    Returns:
        float, float, float, float: FP, FN, TP, TN
    """
    FP = float(np.sum(prediction * (1.0 - groundtruth)))
    FN = float(np.sum((1.0 - prediction) * groundtruth))
    TP = float(np.sum(prediction * groundtruth))
    TN = float(np.sum((1.0 - prediction) * (1.0 - groundtruth)))
    return FP, FN, TP, TN


def precision_score(prediction, groundtruth, err_value=0.0):
    """Positive predictive value (PPV).

    Precision equals the number of true positive voxels divided by the sum of true and false positive voxels.
    True and false positives are computed on soft masks, see ``"numeric_score"``.
    Taken from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/metrics.py

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Precision score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FP) <= 0.0:
        return err_value

    precision = np.divide(TP, TP + FP)
    return precision


def recall_score(prediction, groundtruth, err_value=0.0):
    """True positive rate (TPR).

    Recall equals the number of true positive voxels divided by the sum of true positive and false negative voxels.
    True positive and false negative values are computed on soft masks, see ``"numeric_score"``.
    Taken from: https://github.com/ivadomed/ivadomed/blob/master/ivadomed/metrics.py

    Args:
        prediction (ndarray): First array.
        groundtruth (ndarray): Second array.
        err_value (float): Value returned in case of error.

    Returns:
        float: Recall score.
    """
    FP, FN, TP, TN = numeric_score(prediction, groundtruth)
    if (TP + FN) <= 0.0:
        return err_value
    TPR = np.divide(TP, TP + FN)
    return TPR


def dice_score(prediction, groundtruth):
    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def plot_slices(image, gt, pred):
    """
    Plot the image, ground truth and prediction of the mid-sagittal axial slice
    The orientaion is assumed to RPI
    """

    # bring everything to numpy
    image = image.numpy()
    gt = gt.numpy()
    pred = pred.numpy()

    fig, axs = plt.subplots(1, 3, figsize=(10, 8))
    fig.suptitle('Original Image --> Ground Truth --> Prediction')
    slice = image.shape[2]//2

    axs[0].imshow(image[:, :, slice].T, cmap='gray'); axs[0].axis('off') 
    axs[1].imshow(gt[:, :, slice].T); axs[1].axis('off')
    axs[2].imshow(pred[:, :, slice].T); axs[2].axis('off')

    plt.tight_layout()
    fig.show()
    return fig


class PolyLRScheduler(_LRScheduler):
    """
    Polynomial learning rate scheduler. Taken from:
    https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/lr_scheduler/polylr.py

    """

    def __init__(self, optimizer, initial_lr: float, max_steps: int, exponent: float = 0.9, current_step: int = None):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.max_steps = max_steps
        self.exponent = exponent
        self.ctr = 0
        super().__init__(optimizer, current_step if current_step is not None else -1, False)

    def step(self, current_step=None):
        if current_step is None or current_step == -1:
            current_step = self.ctr
            self.ctr += 1

        new_lr = self.initial_lr * (1 - current_step / self.max_steps) ** self.exponent
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr


if __name__ == "__main__":

    seed = 54
    num_cv_folds = 10
    names_list = FoldGenerator(seed, num_cv_folds, 100).get_fold_names()
    tr_ix, val_tx, te_ix, fold = names_list[0]
    print(len(tr_ix), len(val_tx), len(te_ix))