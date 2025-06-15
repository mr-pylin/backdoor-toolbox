from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    """
    Computes the knowledge distillation loss between a student and teacher model.

    Combines soft (KL-divergence between softmax outputs) and hard (cross-entropy with ground truth) losses.

    Args:
        temperature (float): Temperature scaling factor for softmax. Higher values soften the probability distribution.
        alpha (float): Weighting factor between soft and hard losses.

    Attributes:
        kl_loss (nn.KLDivLoss): KL divergence loss for soft targets.
        ce_loss (nn.CrossEntropyLoss): Cross-entropy loss for hard targets.
    """

    def __init__(self, temperature: float = 1.0, alpha: float = 0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        """
        Compute the distillation loss.

        Args:
            student_logits (Tensor): Output logits from the student model of shape (B, C).
            teacher_logits (Tensor): Output logits from the teacher model of shape (B, C).
            targets (Tensor): Ground truth class indices of shape (B,).

        Returns:
            Tensor: Scalar loss combining soft and hard losses.
        """
        T = self.temperature
        soft_loss = self.kl_loss(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1)) * (
            T * T
        )

        hard_loss = self.ce_loss(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
