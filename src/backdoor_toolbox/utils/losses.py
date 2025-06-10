from torch import nn
import torch.nn.functional as F


class DistillationLoss(nn.Module):
    def __init__(self, temperature=1.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_loss = nn.KLDivLoss(reduction="batchmean")
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, student_logits, teacher_logits, targets):
        T = self.temperature
        soft_loss = self.kl_loss(F.log_softmax(student_logits / T, dim=1), F.softmax(teacher_logits / T, dim=1)) * (
            T * T
        )

        hard_loss = self.ce_loss(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss
