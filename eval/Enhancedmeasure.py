import numpy as np


def emeasure(fm, gt):
    """
    Compute the Enhanced Alignment measure
    Fan D P, Gong C, Cao Y, et al.
    Enhanced-alignment measure for binary foreground map evaluation[J].
    arXiv preprint arXiv:1805.10421, 2018.
    Args:
        fm: Binary foreground map. Type: ndarray
        gt: Binary ground truth. Type: ndarray

    Returns:
        score: The Enhanced alignment score
    """
    fm = np.array(fm)
    gt = np.array(gt)

    if fm.max() > 1:
        fm = fm / 255.0
    if gt.max() > 1:
        gt = gt / 255.0

    gt_binary = gt > 0.5

    dfm = fm.astype(float)
    dgt = gt.astype(float)
    if np.sum(dgt) == 0:
        enhanced_matrix = 1.0 - dfm
    elif np.sum(dgt < 0.5) == 0:
        enhanced_matrix = dfm
    else:
        align_matrix = alignment_term(dfm, dgt)
        enhanced_matrix = enhanced_alignment_term(align_matrix)

    h, w = gt.shape
    score = np.sum(enhanced_matrix) / (w * h - 1 + np.finfo(float).eps)

    return score


def alignment_term(dfm, dgt):
    """
    Compute alignment matrix

    Args:
        dfm: Foreground map in float
        dgt: Ground truth in float

    Returns:
        align_matrix: Alignment matrix
    """
    mu_fm = np.mean(dfm)
    mu_gt = np.mean(dgt)

    align_fm = dfm - mu_fm
    align_gt = dgt - mu_gt

    align_matrix = 2 * (align_gt * align_fm) / (align_gt * align_gt + align_fm * align_fm + np.finfo(float).eps)

    return align_matrix


def enhanced_alignment_term(align_matrix):
    """
    Enhanced Alignment Term function. f(x) = 1/4*(1 + x)^2

    Args:
        align_matrix: Alignment matrix

    Returns:
        enhanced: Enhanced alignment matrix
    """
    enhanced = ((align_matrix + 1) ** 2) / 4
    return enhanced