import numpy as np
import nibabel as nib
import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='Convert softmax probabilites (npz) to niftis')
    parser.add_argument('-i', type=str, required=True, help='Path to the npz file')
    parser.add_argument('-o', type=str, required=True,
                        help='Path to the output nifti which stores the softmax probabilities of the SC segmentation')

    return parser


def main():
    
    args = get_parser().parse_args()

    path_seg_npz = args.i
    path_seg_soft = args.o

    path_seg_bin = path_seg_npz.replace('.npz', '.nii.gz')
    seg_bin_nii = nib.load(path_seg_bin)

    with np.load(path_seg_npz) as data:
        # print(data.files)
        seg_probs = data['probabilities']
        # NOTE (1): because sc seg is a 2-channel prediction (i.e. 0: backgroudn adn 1: sc-seg), the .npz is a 4D array of shape
        # arr: (2, H, W, D). Because background is useless for us, we can just take the 2nd channel from this output 
        # which corresponds to the unbinarized (soft) mask. 
        # NOTE (2): because nnUNet also transposed the matrix, we un-transpose it back to match the (RPI) orientation of our test image
        seg_probs_sc = seg_probs[1].transpose(2, 1, 0)

    seg_probs_c1 = nib.nifti1.Nifti1Image(seg_probs_sc, affine=seg_bin_nii.affine, header=seg_bin_nii.header)

    # nib.save(seg_probs_c0, os.path.join(save_path, f"v3ContrastAgnosticAll_{idx:03d}_probs_c0.nii.gz"))
    nib.save(seg_probs_c1, path_seg_soft)
    

if __name__ == "__main__":
    main()

