import numpy as np
import scipy
from scipy import sparse
from scipy.sparse import linalg
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
        Ds.append(D)

        M = new_M
        k += k_step

    return clip_and_convert_to_uint8(M), Ds

def clip_and_convert_to_uint8(img):
    np.clip(img, 0, 255).astype('uint8')

def envelope_bound_interpolation(M, extrema_mask):
    flat_M = M.ravel()
    flat_extrema_mask = extrema_mask.ravel()

    pixel_num = flat_M.size
    neighbor_k = 3
    extrema_padded_mask, padding = pad_zeros(extrema_mask, neighbor_k)
    padded_constraint_idx = np.nonzero(extrema_padded_mask.ravel())[0]
    constraint_idx = np.nonzero(extrema_mask.ravel())[0]
    constraint_num = padded_constraint_idx.size # same size with constraint_idx


    b = np.zeros(pixel_num + constraint_num)

    #fill the optimization part
    #since the implementation difficulty, w must introduces redundant padded variables
    A_optimize = compute_A_optimize(M)

    #fill the constriant part
    A_col_idx = np.arange(constraint_num)
    A_row_idx = padded_constraint_idx
    A_constraint_fill = np.ones(constraint_num)

    #it seems not suitable to use bsr_matrix, since it throws runtime error
    A_constraint = sparse.csr_matrix((A_constraint_fill, (A_col_idx, A_row_idx)), shape=(constraint_num, A_optimize.shape[1]))
    b[-constraint_num:] = flat_M[constraint_idx]

    A = sparse.vstack((A_optimize, A_constraint))

    E = linalg.lsmr(A, b, maxiter=1000)[0]
    return E.reshape(extrema_padded_mask.shape)[padding:-padding, padding:-padding]

EPSILON = 1e-9
def compute_A_optimize(M, neighbor_k=3):
    pixel_num = M.size
    center_idx = (neighbor_k * neighbor_k) // 2
    padded_M = pad_img_for_conv(M, neighbor_k)
    conv_idx_vol = conv_idx_volume(padded_M, M.shape, neighbor_k)

    neighbor_vol = np.take(padded_M, conv_idx_vol)
    variance = np.var(neighbor_vol, 0)  # note the variance includes the padded pixels
    variance[variance < EPSILON] = EPSILON

    A_col_idx = conv_idx_vol.ravel()
    A_row_idx = np.tile(np.arange(pixel_num)[:, np.newaxis], [1, conv_idx_vol.shape[0]]).ravel() # conv_idx_vol.shape[0] == k**2
    fill_A_2d = -(np.exp(-((M - neighbor_vol) ** 2) / variance)) # compute w(r, s), and times -1
    fill_A_2d[center_idx] = 1 #set the center pixel's coefficient to 1

    #note, w here contains pixels from padding, so it introduces several variables to be solved,
    #since removeing these variable is hard in the sparse matrix form
    A = sparse.bsr_matrix((fill_A_2d.ravel(), (A_row_idx, A_col_idx)), shape=(pixel_num, padded_M.size))
    return A

def local_extrema(img, k):
    vol = conv_volume(img, k)
    center_idx = (k * k) // 2

    lt_cent_statistics = np.sum(vol < vol[center_idx], axis=0)
    gt_cent_statistics = np.sum(vol > vol[center_idx], axis=0)

    EXTREMA_CRITERIA = k - 1
    return gt_cent_statistics <= EXTREMA_CRITERIA, lt_cent_statistics <= EXTREMA_CRITERIA

def conv_volume(img, k):
    padded_img = pad_img_for_conv(img, k)
    volume = np.take(padded_img, conv_idx_volume(padded_img, img.shape, k))
    return volume

def pad_img_for_conv(img, k):
    padding = k // 2
    padded_img = np.pad(img, padding, 'reflect')

    return padded_img

def pad_zeros(img, k):
    padding = k // 2
    padded_img = np.pad(img, padding, 'constant', constant_values=0)
    return padded_img, padding

def conv_idx_volume(padded_img, img_shape, k):
    H, W = padded_img.shape

    # start_idx_slice = np.arange(H * W).reshape((H, W))[:-k + 1, :-k + 1]
    # use broadcast trick: column vector + row vector = matrix
    start_idx_slice = np.arange(img_shape[0])[:, np.newaxis] * W + np.arange(img_shape[1])
    conv_idx = np.arange(k)[:, np.newaxis] * W + np.arange(k)

    conv_idx_3d = conv_idx.ravel()[:, np.newaxis, np.newaxis]

    # use broadcast trick: 3d column vector + matrix = volume
    return conv_idx_3d + start_idx_slice