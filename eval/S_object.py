import numpy as np


def s_object(prediction, gt):
    """
    Compute the object similarity between foreground maps and ground truth
    Fan D P, Cheng M M, Liu Y, et al.
    Structure-measure: A new way to evaluate foreground maps
    Proceedings of the IEEE international conference on computer vision. 2017: 4548-4557.
    Args:
        prediction: Binary/Non-binary foreground map with values in [0,1]. Type: float
        gt: Binary ground truth. Type: bool

    Returns:
        Q: The object similarity score
    """
    # Compute similarity of foreground in object level
    prediction_fg = prediction.copy()
    prediction_fg[~gt] = 0
    o_fg = object_score(prediction_fg, gt)

    # Compute similarity of background
    prediction_bg = 1.0 - prediction
    prediction_bg[gt] = 0
    o_bg = object_score(prediction_bg, ~gt)

    # Combine foreground and background measures
    u = np.mean(gt)
    q = u * o_fg + (1 - u) * o_bg

    return q


def object_score(prediction, gt):
    """
    Calculate object score for either foreground or background

    Args:
        prediction: Prediction map
        gt: Ground truth

    Returns:
        score: Object score
    """
    # Input validation
    if prediction.size == 0:
        return 0

    if prediction.dtype != np.float64:
        prediction = prediction.astype(np.float64)

    if not isinstance(prediction, np.ndarray) or prediction.dtype != np.float64:
        raise TypeError('prediction should be of type: float64')

    if np.max(prediction) > 1 or np.min(prediction) < 0:
        raise ValueError('prediction should be in the range of [0 1]')

    if not np.issubdtype(gt.dtype, np.bool_):
        raise TypeError('GT should be of type: bool')

    # Compute mean of foreground/background in prediction
    x = np.mean(prediction[gt])

    # Compute standard deviation of foreground/background in prediction
    sigma_x = np.std(prediction[gt])

    # Calculate score
    eps = np.finfo(float).eps
    score = 2.0 * x / (x ** 2 + 1.0 + sigma_x + eps)

    return score