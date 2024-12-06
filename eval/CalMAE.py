import numpy as np

def cal_mae(smap, gt_img):
    """
    Code Author: Wangjiang Zhu
    Email: wangjiang88119@gmail.com
    Date: 3/24/2014
    Calculate Mean Absolute Error between saliency map and ground truth image
    Args:
        smap: saliency map
        gt_img: ground truth image
    Returns:
        mae: mean absolute error
    """
    if smap.shape != gt_img.shape:
        raise ValueError('Saliency map and ground truth image have different sizes!')

    if not gt_img.dtype == bool:
        gt_img = gt_img > 128

    if smap.dtype != np.float32 and smap.dtype != np.float64:
        smap = smap.astype(float)
        if smap.max() > 1.0:
            smap = smap / 255.0

    fg_pixels = smap[gt_img]
    fg_err_sum = len(fg_pixels) - np.sum(fg_pixels)
    bg_err_sum = np.sum(smap[~gt_img])

    mae = (fg_err_sum + bg_err_sum) / gt_img.size

    return mae