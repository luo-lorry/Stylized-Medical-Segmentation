import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import convolve2d


def gaussian_kernel(size, sigma):
    """
    Generate a 2D Gaussian kernel

    Args:
        size: Tuple of (height, width) or single integer for square kernel
        sigma: Standard deviation of Gaussian

    Returns:
        2D Gaussian kernel
    """
    if isinstance(size, int):
        size = (size, size)

    y, x = np.meshgrid(np.linspace(-size[0] // 2, size[0] // 2, size[0]),
                       np.linspace(-size[1] // 2, size[1] // 2, size[1]))

    gaussian = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return gaussian / gaussian.sum()


def weighted_f_measure(fg, gt):
    """
    Compute the Weighted F-beta measure
    Margolin R, Zelnik-Manor L, Tal A.
    How to evaluate foreground maps?
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2014: 248-255.
    Args:
        fg: Binary/Non-binary foreground map with values in [0,1]. Type: float
        gt: Binary ground truth. Type: bool

    Returns:
        Q: The Weighted F-beta score
    """
    # Input validation
    if not isinstance(fg, np.ndarray) or fg.dtype != np.float64:
        raise TypeError('FG should be of type: float64')
    if np.max(fg) > 1 or np.min(fg) < 0:
        raise ValueError('FG should be in the range of [0 1]')
    if not np.issubdtype(gt.dtype, np.bool_):
        raise TypeError('GT should be of type: bool')

    # Convert ground truth to double
    dgt = gt.astype(float)

    # Calculate error
    E = np.abs(fg - dgt)

    # Calculate distance transform
    Dst = distance_transform_edt(~gt)
    IDXT = np.zeros_like(gt)

    # Get indices of nearest true pixels for false pixels
    if np.any(~gt):
        idx_true = np.nonzero(gt)
        idx_false = np.nonzero(~gt)

        # Calculate distances between all false and true pixels
        distances = np.sqrt((idx_false[0][:, None] - idx_true[0][None, :]) ** 2 +
                            (idx_false[1][:, None] - idx_true[1][None, :]) ** 2)

        # Get indices of minimum distances
        nearest_true = np.argmin(distances, axis=1)

        # Convert linear indices to pixel positions
        IDXT[~gt] = np.ravel_multi_index((idx_true[0][nearest_true],
                                          idx_true[1][nearest_true]),
                                         gt.shape)

    # Pixel dependency
    K = gaussian_kernel(7, 5)
    Et = E.copy()
    Et[~gt] = Et.flat[IDXT[~gt]]
    EA = convolve2d(Et, K, mode='same')

    MIN_E_EA = E.copy()
    mask = gt & (EA < E)
    MIN_E_EA[mask] = EA[mask]

    # Pixel importance
    B = np.ones_like(gt, dtype=float)
    B[~gt] = 2.0 - np.exp(np.log(1 - 0.5) / 5 * Dst[~gt])

    Ew = MIN_E_EA * B

    # Calculate metrics
    TPw = np.sum(dgt) - np.sum(Ew[gt])
    FPw = np.sum(Ew[~gt])

    # Weighted Recall
    R = 1 - np.mean(Ew[gt])

    # Weighted Precision
    eps = np.finfo(float).eps
    P = TPw / (eps + TPw + FPw)

    # Calculate Q (Beta=1)
    Q = (2 * R * P) / (eps + R + P)

    return Q