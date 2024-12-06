import numpy as np


def s_region(prediction, gt):
    """
    Compute region similarity between foreground map and ground truth

    Args:
        prediction: Binary/Non-binary foreground map with values in [0,1]. Type: float
        gt: Binary ground truth. Type: bool

    Returns:
        Q: Region similarity score
    """
    # Find centroid of GT
    x, y = centroid(gt)

    # Divide GT into 4 regions
    gt_1, gt_2, gt_3, gt_4, w1, w2, w3, w4 = divide_gt(gt, x, y)

    # Divide prediction into 4 regions
    pred_1, pred_2, pred_3, pred_4 = divide_prediction(prediction, x, y)

    # Compute ssim score for each region
    q1 = ssim(pred_1, gt_1)
    q2 = ssim(pred_2, gt_2)
    q3 = ssim(pred_3, gt_3)
    q4 = ssim(pred_4, gt_4)

    # Sum weighted scores
    q = w1 * q1 + w2 * q2 + w3 * q3 + w4 * q4

    return q


def centroid(gt):
    """
    Compute centroid coordinates of ground truth
    """
    rows, cols = gt.shape

    if np.sum(gt) == 0:
        return round(cols / 2), round(rows / 2)

    total = np.sum(gt)
    x_coords = np.arange(1, cols + 1)
    y_coords = np.arange(1, rows + 1)

    x = round(np.sum(np.sum(gt, axis=0) * x_coords) / total)
    y = round(np.sum(np.sum(gt, axis=1) * y_coords) / total)

    return x, y


def divide_gt(gt, x, y):
    """
    Divide ground truth into 4 regions and compute weights
    """
    height, width = gt.shape
    area = width * height

    # Copy 4 regions
    lt = gt[0:y, 0:x]
    rt = gt[0:y, x:width]
    lb = gt[y:height, 0:x]
    rb = gt[y:height, x:width]

    # Compute weights
    w1 = (x * y) / area
    w2 = ((width - x) * y) / area
    w3 = (x * (height - y)) / area
    w4 = 1.0 - w1 - w2 - w3

    return lt, rt, lb, rb, w1, w2, w3, w4


def divide_prediction(prediction, x, y):
    """
    Divide prediction into 4 regions
    """
    height, width = prediction.shape

    lt = prediction[0:y, 0:x]
    rt = prediction[0:y, x:width]
    lb = prediction[y:height, 0:x]
    rb = prediction[y:height, x:width]

    return lt, rt, lb, rb


def ssim(prediction, gt):
    """
    Compute structural similarity between prediction and ground truth regions
    """
    d_gt = gt.astype(float)

    height, width = prediction.shape
    n = width * height
    eps = np.finfo(float).eps

    # Compute means
    x = np.mean(prediction)
    y = np.mean(d_gt)

    # Compute variances
    sigma_x2 = np.sum((prediction - x) ** 2) / (n - 1 + eps)
    sigma_y2 = np.sum((d_gt - y) ** 2) / (n - 1 + eps)

    # Compute covariance
    sigma_xy = np.sum((prediction - x) * (d_gt - y)) / (n - 1 + eps)

    # Compute SSIM
    alpha = 4 * x * y * sigma_xy
    beta = (x ** 2 + y ** 2) * (sigma_x2 + sigma_y2)

    if alpha != 0:
        q = alpha / (beta + eps)
    elif alpha == 0 and beta == 0:
        q = 1.0
    else:
        q = 0

    return q