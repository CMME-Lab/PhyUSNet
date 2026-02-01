import numpy as np
from scipy.spatial.distance import directed_hausdorff
from tqdm import tqdm
import cv2
import torch

def compute_hd95(pred_mask, gt_mask):
    """
    Compute the 95th percentile of the Hausdorff Distance (HD95) between two binary masks.

    Args:
        pred_mask (torch.Tensor): Predicted binary mask, shape (H, W) or (1, H, W)
        gt_mask (torch.Tensor): Ground truth binary mask, shape (H, W) or (1, H, W)

    Returns:
        hd95 (float): The 95th percentile of the Hausdorff Distance
    """
    assert pred_mask.shape == gt_mask.shape, "Masks must have the same shape"
    # print(pred_mask.shape, gt_mask.shape)
    # Convert to numpy and binarize
    pred_mask = pred_mask.squeeze().cpu().numpy().astype(np.uint8)  # Shape (H, W)
    gt_mask = gt_mask.squeeze().cpu().numpy().astype(np.uint8)  # Shape (H, W)
    if pred_mask.shape[0] != 256:
        pred_mask = cv2.resize(pred_mask, (256, 256))
        gt_mask = cv2.resize(gt_mask, (256, 256))
    # Get coordinates of foreground pixels (x, y)
    pred_coords = np.column_stack(np.where(pred_mask > 0))
    gt_coords = np.column_stack(np.where(gt_mask > 0))

    # If either mask is empty, assign a penalty value
    if len(pred_coords) == 0 or len(gt_coords) == 0:
        # print("Empty mask detected")
        return 50  # Reasonable high penalty for empty masks

    # Compute directed Hausdorff distances
    hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
    hd2 = directed_hausdorff(gt_coords, pred_coords)[0]

    # Compute HD95 (95th percentile of all Hausdorff distances)
    hd95 = np.percentile([hd1, hd2], 95)

    return hd95


def get_metrics(outputs, targets, threshold=0.5):
    """
    Calculates evaluation metrics for binary segmentation per image and averages over the batch.
    Args:
        outputs (torch.Tensor): Model predictions, expected to be probabilities in [0,1], shape [batch_size, 1, H, W]
        targets (torch.Tensor): Ground truth labels, shape [batch_size, 1, H, W]
    Returns:
        metrics_dict (dict): Dictionary containing averaged 'dice', 'iou', 'precision', 'recall', 'accuracy'
    """
    epsilon = 1e-7  # Small constant to avoid division by zero

    # Normalize target shape to [B, H, W]
    if targets.dim() == 4 and targets.size(1) == 1:
        targets_ = targets[:, 0]
    elif targets.dim() == 3:
        targets_ = targets
    else:
        # Fallback: try to squeeze channel
        targets_ = targets.squeeze(1) if targets.dim() == 4 else targets

    # Convert outputs to binary mask [B, H, W]
    if outputs.dim() == 4 and outputs.size(1) > 1:
        # Multiclass: pick foreground class (1) vs background (0)
        probs = torch.softmax(outputs, dim=1)
        outputs_ = (probs[:, 1] > 0.5).float()
    else:
        # Binary: ensure probabilities and threshold
        logits_or_probs = outputs[:, 0] if outputs.dim() == 4 else outputs
        # If clearly in [0,1], treat as probabilities; else apply sigmoid
        vmin = float(logits_or_probs.min().detach().cpu())
        vmax = float(logits_or_probs.max().detach().cpu())
        if vmin >= -1e-6 and vmax <= 1.0 + 1e-6:
            probs = logits_or_probs
        else:
            probs = torch.sigmoid(logits_or_probs)
        outputs_ = (probs > threshold).float()

    targets_bin = (targets_ > threshold).float()
    outputs_bin = outputs_.float()

    # Flatten per-sample for vectorized confusion counts
    B = outputs_bin.size(0)
    pred_flat = outputs_bin.view(B, -1)
    true_flat = targets_bin.view(B, -1)

    TP = (pred_flat * true_flat).sum(dim=1)
    FP = (pred_flat * (1.0 - true_flat)).sum(dim=1)
    TN = ((1.0 - pred_flat) * (1.0 - true_flat)).sum(dim=1)
    FN = ((1.0 - pred_flat) * true_flat).sum(dim=1)

    dice_per = (2 * TP + epsilon) / (2 * TP + FP + FN + epsilon)
    iou_per = (TP + epsilon) / (TP + FP + FN + epsilon)
    precision_per = (TP + epsilon) / (TP + FP + epsilon)
    recall_per = (TP + epsilon) / (TP + FN + epsilon)
    accuracy_per = (TP + TN + epsilon) / (TP + TN + FP + FN + epsilon)

    # HD95 needs per-sample computation
    hd95_vals = []
    for i in range(B):
        hd95_i = compute_hd95(outputs_bin[i], targets_bin[i])
        hd95_vals.append(hd95_i)

    # FPR and TPR per-sample
    fpr_per = (FP) / (FP + TN + epsilon)
    tpr_per = (TP) / (TP + FN + epsilon)

    # Convert to scalar means
    dice_scores = float(dice_per.mean().detach().cpu())
    iou_scores = float(iou_per.mean().detach().cpu())
    precision_scores = float(precision_per.mean().detach().cpu())
    recall_scores = float(recall_per.mean().detach().cpu())
    accuracy_scores = float(accuracy_per.mean().detach().cpu())
    hd95_scores = float(np.mean(hd95_vals)) if len(hd95_vals) > 0 else 0.0
    fpr = float(fpr_per.mean().detach().cpu())
    tpr = float(tpr_per.mean().detach().cpu())

    metrics_dict = {
        "dice": dice_scores,
        "iou": iou_scores,
        "precision": precision_scores,
        "recall": recall_scores,
        "accuracy": accuracy_scores,
        "hd95": hd95_scores,
        "fpr": fpr,
        "tpr": tpr,
    }

    return metrics_dict


