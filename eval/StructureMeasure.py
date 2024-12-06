import numpy as np
from eval.S_object import s_object
from eval.S_region import s_region


def structure_measure(prediction, gt):
    """
    Compute structure measure similarity between foreground map and ground truth

    Args:
        prediction: Binary/Non binary foreground map with values in [0,1]. Type: float
        gt: Binary ground truth. Type: bool

    Returns:
        Q: Structure measure similarity score
    """
    # Input validation
    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        raise TypeError('The prediction should be float64 type')

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError('The prediction should be in the range of [0 1]')

    if not np.issubdtype(gt.dtype, np.bool_):
        raise TypeError('GT should be bool type')

    # Calculate mean of ground truth
    y = np.mean(gt)

    if y == 0:  # GT is completely black
        x = np.mean(prediction)
        q = 1.0 - x  # Only calculate area of intersection

    elif y == 1:  # GT is completely white
        x = np.mean(prediction)
        q = x  # Only calculate area of intersection

    else:
        alpha = 0.5
        q = alpha * s_object(prediction, gt) + (1 - alpha) * s_region(prediction, gt)
        if q < 0:
            q = 0

    return q