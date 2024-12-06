import numpy as np


def fmeasure_calu(smap, gt_map, gt_size, threshold):
    """
    Calculate various metrics (Precision, Recall, Specificity, Dice, F-measure, IoU)

    Args:
        smap: Saliency map
        gt_map: Ground truth map
        gt_size: Size of ground truth
        threshold: Threshold value for binarization

    Returns:
        PreFtem: Precision
        RecallFtem: Recall
        SpecifTem: Specificity
        Dice: Dice coefficient
        FmeasureF: F-measure
        IoU: Intersection over Union
    """
    # Threshold check
    if threshold > 1:
        threshold = 1

    # Create binary prediction
    label3 = np.zeros(gt_size)
    label3[smap >= threshold] = 1

    # Calculate basic counts
    num_rec = np.sum(label3 == 1)  # FP+TP
    num_no_rec = np.sum(label3 == 0)  # FN+TN
    label_and = np.logical_and(label3, gt_map)
    num_and = np.sum(label_and)  # TP
    num_obj = np.sum(gt_map)  # TP+FN
    num_pred = np.sum(label3)  # FP+TP

    # Calculate TP, FP, FN, TN
    fn = num_obj - num_and
    fp = num_rec - num_and
    tn = num_no_rec - fn

    # Handle case when no intersection
    if num_and == 0:
        return 0, 0, 0, 0, 0, 0

    # Calculate metrics
    iou = num_and / (fn + num_rec)  # TP/(FN+TP+FP)
    pre_ftem = num_and / num_rec  # Precision
    recall_ftem = num_and / num_obj  # Recall
    specif_tem = tn / (tn + fp)  # Specificity
    dice = 2 * num_and / (num_obj + num_pred)  # Dice coefficient

    # Calculate F-measure (beta = 1.0)
    fmeasure_f = (2.0 * pre_ftem * recall_ftem) / (pre_ftem + recall_ftem)

    return pre_ftem, recall_ftem, specif_tem, dice, fmeasure_f, iou