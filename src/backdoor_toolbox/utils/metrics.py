import torch
from torchmetrics import Metric


class AttackSuccessRate(Metric):
    """
    Metric to calculate the attack success rate for backdoor attacks.

    Args:
        target_index (int): The target class index for the backdoor attack.

    Attributes:
        target_class (int): The target class for the attack.
        success (torch.Tensor): Number of successful attacks.
        total (torch.Tensor): Total number of samples evaluated.
    """

    target_class: int
    success: torch.Tensor
    total: torch.Tensor

    def __init__(self, target_index: int):
        """
        Initializes the AttackSuccessRate metric.

        Args:
            target_index (Optional[int]): The target class for the backdoor attack. If None, computes accuracy for all samples.
        """
        super().__init__()
        self.target_class = target_index
        self.add_state("success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, poison_mask: torch.Tensor | None) -> None:
        """
        Update the metric with new predictions and optional poison mask.

        Args:
            preds (torch.Tensor): Model predictions with shape (batch_size, num_classes).
            poison_mask (Optional[torch.Tensor]): Boolean mask for poisoned samples.
        """
        if poison_mask is not None:
            preds = preds[poison_mask]

        preds = preds.argmax(dim=-1)
        self.success += (preds == self.target_class).sum()
        self.total += len(preds)

    def compute(self) -> torch.Tensor:
        """
        Compute the attack success rate.

        Returns:
            torch.Tensor: Attack success rate (success / total).
        """
        return self.success.float() / self.total


class CleanDataAccuracy(Metric):
    """
    Metric to calculate accuracy on clean data.

    Attributes:
        correct (torch.Tensor): Number of correct predictions on clean samples.
        total (torch.Tensor): Total number of clean samples.
    """

    correct: torch.Tensor
    total: torch.Tensor

    def __init__(self):
        """Initializes the CleanDataAccuracy metric."""
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, clean_mask: torch.Tensor | None) -> None:
        """
        Update the metric with predictions, targets, and optional clean mask.

        Args:
            preds (torch.Tensor): Model predictions with shape (batch_size, num_classes).
            targets (torch.Tensor): Ground truth labels with shape (batch_size,).
            clean_mask (Optional[torch.Tensor]): Boolean mask for clean samples.
        """
        if clean_mask is not None:
            preds = preds[clean_mask]

        preds = preds.argmax(dim=-1)
        targets = targets[clean_mask]

        self.correct += (preds == targets).sum()
        self.total += len(preds)

    def compute(self) -> torch.Tensor:
        """
        Compute the accuracy on clean data.

        Returns:
            torch.Tensor: Clean data accuracy (correct / total).
        """
        return self.correct.float() / self.total


if __name__ == "__main__":

    # simulated data
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    targets = torch.tensor([1, 0, 1])
    clean_mask = torch.tensor([True, False, True])
    poison_mask = ~clean_mask

    # instantiate metrics
    asr = AttackSuccessRate(target_index=0)
    cda = CleanDataAccuracy()

    # update metrics
    asr.update(preds, poison_mask)
    cda.update(preds, targets, clean_mask)

    # compute results
    print(f"ASR: {asr.compute().item()}")
    print(f"CDA: {cda.compute().item()}")
