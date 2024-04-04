import nibabel as nib
import numpy as np
import argparse


def get_parser():

    parser = argparse.ArgumentParser(description="Compute the dice score for each slice in the volume")

    parser.add_argument("--path-pred", type=str, help="Path to the predicted segmentation", required=True)
    parser.add_argument("--path-gt", type=str, help="Path to the ground truth segmentation", required=True)

    return parser


def compute_slicewise_dice(pred_path, gt_path):
    """
    Compute the dice score for each slice in the volume
    """
    pred = nib.load(pred_path).get_fdata()
    gt = nib.load(gt_path).get_fdata()
    num_slices = pred.shape[2]
    dice_scores = np.zeros(num_slices)
    for slice_idx in range(num_slices):
        pred_slice = pred[:, :, slice_idx]
        gt_slice = gt[:, :, slice_idx]
        dice_scores[slice_idx] = dice_score(pred_slice, gt_slice)

    return np.mean(dice_scores)


def dice_score(prediction, groundtruth):
    """
    Adapted from MetricsReloaded 
    https://github.com/Project-MONAI/MetricsReloaded/blob/main/MetricsReloaded/metrics/pairwise_measures.py#L396
    """

    prediction = np.asarray(prediction, dtype=np.float32)
    groundtruth = np.asarray(groundtruth, dtype=np.float32)
    tp_map = np.asarray((prediction + groundtruth) > 1.0, dtype=np.float32)
    tp = np.sum(tp_map)

    numerator = 2*tp
    denominator = np.sum(prediction) + np.sum(groundtruth)
    if denominator == 0:
        # print("Both Prediction and Reference are empty - set to 1 as correct solution even if not defined")
        return 1
    else:
        return numerator / denominator


def dice_score_og(pred_path, gt_path):

    prediction = nib.load(pred_path).get_fdata()
    groundtruth = nib.load(gt_path).get_fdata()

    smooth = 1.
    numer = (prediction * groundtruth).sum()
    denor = (prediction + groundtruth).sum()
    # loss = (2 * numer + self.smooth) / (denor + self.smooth)
    dice = (2 * numer + smooth) / (denor + smooth)
    return dice


def main():

    parser = get_parser()
    args = parser.parse_args()

    dice = compute_slicewise_dice(args.path_pred, args.path_gt)
    print(dice)
    # print(f"Slice-wise dice score: {dice}")
    # dice = dice_score_og(args.path_pred, args.path_gt)
    # print(f"Original dice score: {dice}")

    # dice = compute_slicewise_dice(args.path_pred, args.path_gt)
    # print(dice)


if __name__ == "__main__":
    main()
    
    # dice = compute_slicewise_dice(pred_path, gt_path)
    # print(f"Slice-wise dice score: {dice}")
    # dice = dice_score_og(pred_path, gt_path)
    # print(f"Original dice score: {dice}")