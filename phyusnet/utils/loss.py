'''
Loss Function for Binary Segmentation 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

def dice_loss_binary(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    from_logits: bool = True,
    eps: float = 1e-7,
    smooth: float = 0.0,
    p: float = 1.0,
    reduction: str = "mean",
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Dice loss for binary segmentation with inputs of shape (N, 1, H, W).

    Args:
        preds: Model outputs. Shape (N, 1, H, W). Logits if from_logits is True.
        targets: Binary ground truth in {0, 1}. Shape (N, 1, H, W) or (N, H, W).
        from_logits: Apply sigmoid if True. Otherwise treat preds as probabilities.
        eps: Small constant for numerical stability.
        smooth: Additive smoothing in numerator and denominator.
        p: Exponent on probabilities and targets in denominator. Use 1 for standard Dice. Use 2 for a Tversky style variant.
        reduction: One of "none", "mean", "sum".
        ignore_index: Label value in targets to ignore. Use None to disable.

    Returns:
        Tensor with the Dice loss. Reduced according to the reduction argument.
    """
    if targets.dim() == 3:
        targets = targets.unsqueeze(1)
    if preds.shape != targets.shape:
        raise ValueError(f"Shape mismatch: preds {preds.shape} vs targets {targets.shape}")

    if from_logits:
        probs = torch.sigmoid(preds)
    else:
        probs = preds.clamp(min=0.0, max=1.0)

    # Optional ignore mask
    if ignore_index is not None:
        valid_mask = (targets != ignore_index).float()
        # Replace ignored target positions with zero to keep multiplications safe
        targets = torch.where(valid_mask > 0, targets, torch.zeros_like(targets))
        probs = probs * valid_mask
    else:
        valid_mask = torch.ones_like(targets)

    # Flatten per sample over spatial dimensions
    N = preds.shape[0]
    probs_f = probs.view(N, -1)
    targets_f = targets.view(N, -1)
    mask_f = valid_mask.view(N, -1)

    # Intersection and denominator with optional power p
    # Standard Dice uses p = 1
    intersection = (probs_f * targets_f).sum(dim=1)

    denom = (probs_f.pow(p) + targets_f.pow(p)).sum(dim=1)

    dice = (2.0 * intersection + smooth) / (denom + smooth + eps)
    loss = 1.0 - dice

    if reduction == "mean":
        # Weight the mean by the number of valid pixels per sample if an ignore mask is present
        if ignore_index is not None:
            valid_counts = mask_f.sum(dim=1).clamp_min(1.0)
            # Use unweighted mean across batch as is common, or weight by valid fraction
            # Here we use the unweighted mean across samples
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    elif reduction == "none":
        return loss
    else:
        raise ValueError(f"Invalid reduction='{reduction}'. Use 'none', 'mean', or 'sum'.")


class DiceLossBinary(nn.Module):
    def __init__(
        self,
        from_logits: bool = True,
        eps: float = 1e-7,
        smooth: float = 0.0,
        p: float = 1.0,
        reduction: str = "mean",
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        self.from_logits = from_logits
        self.eps = eps
        self.smooth = smooth
        self.p = p
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return dice_loss_binary(
            preds,
            targets,
            from_logits=self.from_logits,
            eps=self.eps,
            smooth=self.smooth,
            p=self.p,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

class CombinedLoss(nn.Module):
    """
    Combined loss function: lambda * DiceLoss + (1 - lambda) * CrossEntropyLoss
    """

    def __init__(self, lambda_weight=0.5, smooth=1e-8, from_logits=True):
        super(CombinedLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.dice_loss = DiceLossBinary(smooth=smooth,from_logits = from_logits)
        # self.ce_loss = nn.CrossEntropyLoss()
        if from_logits:
            self.ce_loss = nn.BCEWithLogitsLoss()
        else:
            self.ce_loss = nn.BCELoss()

    def forward(self, predictions, targets):
        """
        Args:
            predictions: Tensor of shape (N, C, H, W) - raw logits from model
            targets: Tensor of shape (N,1, H, W) - ground truth labels
        """
        dice_loss_value = self.dice_loss(predictions, targets.float())
        ce_loss_value = self.ce_loss(predictions.squeeze(1), targets.float())

        combined_loss = (
            self.lambda_weight * dice_loss_value
            + (1 - self.lambda_weight) * ce_loss_value
        )

        return combined_loss, dice_loss_value, ce_loss_value
