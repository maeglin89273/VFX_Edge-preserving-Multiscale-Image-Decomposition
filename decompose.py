import numpy as np
import utils
def edge_preserving_decompose(img, iter=4, k=3, k_step=8):
    M = img.astype(float)
    Ds = []
    for i in range(iter):
        min_mask, max_mask = local_extrema(M, k)
        envelope_bottom = envelope_bound_interpolation(M, min_mask)
        envelope_top = envelope_bound_interpolation(M, max_mask)
        new_M = (envelope_top + envelope_bottom) / 2
        D = M - new_M
        Ds.append(clip_and_convert_to_uint8(D))

        M = new_M
        k += k_step

    return clip_and_convert_to_uint8(M), Ds

def clip_and_convert_to_uint8(img):
    np.clip(img, 0, 255).astype('uint8')

def envelope_bound_interpolation(M, extrema_mask):
    pass

def local_extrema(img, k):
    vol = conv_volume(img, k)
    center_idx = (k * k) // 2

    lt_cent_statistics = np.sum(vol < vol[center_idx], axis=0)
    gt_cent_statistics = np.sum(vol > vol[center_idx], axis=0)

    EXTREMA_CRITERIA = k - 1
    return gt_cent_statistics <= EXTREMA_CRITERIA, lt_cent_statistics <= EXTREMA_CRITERIA


def conv_volume(img, k):

    padding = k // 2
    padded_img = np.pad(img, padding, 'reflect')

    H, W = padded_img.shape



    # start_idx_slice = np.arange(H * W).reshape((H, W))[:-k + 1, :-k + 1]
    # use broadcast trick: column vector + row vector = matrix
    start_idx_slice = np.arange(img.shape[0])[:, np.newaxis] * W + np.arange(img.shape[1])
    conv_idx = np.arange(k)[:, np.newaxis] * W + np.arange(k)

    conv_idx_3d = conv_idx.ravel()[:, np.newaxis, np.newaxis]

    # use broadcast trick: 3d column vector + matrix = volume
    volume = np.take(padded_img, conv_idx_3d + start_idx_slice)
    return volume