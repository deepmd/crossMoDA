import torch
from monai.metrics import compute_meandice, compute_average_surface_distance, do_metric_reduction
from monai.networks import one_hot


def compute_f1_score(logits, labels, multilabel=False, threshold=0.5):
    if multilabel:
        pred_probs = torch.sigmoid(logits)
        y_pred, y = (pred_probs > threshold).float(), labels
    else:
        y_pred, y = _get_one_hot(logits, labels)

    true_positives = torch.sum(y_pred * y, dim=0)
    all_preds = torch.sum(y_pred, dim=0)
    all_trues = torch.sum(y, dim=0)
    per_class_f1_scores = 2 * true_positives / (all_preds + all_trues)
    macro_avg_f1_score = torch.mean(per_class_f1_scores)

    return macro_avg_f1_score


def compute_dice_score(logits, labels, include_background=True):
    """Computes Dice Score metric per class
    """
    y_pred, y = _get_one_hot(logits, labels)
    dsc = compute_meandice(y_pred, y, include_background=include_background)
    return do_metric_reduction(dsc, reduction="mean_batch")


def compute_average_symmetric_surface_distance(logits, labels, include_background=True):
    """Computes Average Symmetric Surface Distance metric per class
    """
    y_pred, y = _get_one_hot(logits, labels)
    assd = compute_average_surface_distance(y_pred, y, symmetric=True, include_background=include_background)
    return do_metric_reduction(assd, reduction="mean_batch")


def _get_one_hot(logits, labels):
    n_classes = logits.shape[1]
    pred_probs = torch.softmax(logits, dim=1)
    y_pred = torch.argmax(pred_probs, dim=1, keepdim=True)
    y_pred = one_hot(y_pred, n_classes, dim=1)
    y = one_hot(labels, n_classes, dim=1)
    return y_pred, y
