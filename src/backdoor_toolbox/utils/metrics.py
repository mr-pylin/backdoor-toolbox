import torch
from torchmetrics import Metric


class AttackSuccessRate(Metric):
    """
    Metric to calculate the attack success rate for backdoor attacks.

    Attributes:
        target_class (Optional[int]): The target class for the attack. If None, the metric will compute the overall accuracy.
    """

    def __init__(self, target_index: int | None = None):
        """
        Initializes the AttackSuccessRate metric.

        Args:
            target_index (Optional[int]): The target class for the backdoor attack. If None, computes accuracy for all samples.
        """
        super().__init__()
        self.target_class = target_index
        self.add_state("success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, poison_mask: torch.Tensor | None) -> None:
        """
        Updates the metric state with predictions, targets, and optional poison mask.

        Args:
            preds (torch.Tensor): The model predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): The ground truth labels of shape (batch_size).
            poison_mask (Optional[torch.Tensor]): A binary mask indicating poisoned samples. If None, all samples are used.
        """
        if poison_mask is not None:
            preds = preds[poison_mask]

        preds = preds.argmax(dim=-1)

        if self.target_class is not None:
            self.success += (preds == self.target_class).sum()
        else:
            targets = targets[poison_mask]
            self.success += (preds == targets).sum()

        self.total += len(preds)

    def compute(self) -> torch.Tensor:
        """
        Computes the attack success rate.

        Returns:
            torch.Tensor: The attack success rate (success/total).
        """
        return self.success.float() / self.total


class CleanDataAccuracy(Metric):
    """
    Metric to calculate the accuracy on clean data.

    Attributes:
        correct (torch.Tensor): The number of correct predictions on clean data.
        total (torch.Tensor): The total number of samples in clean data.
    """

    def __init__(self):
        """
        Initializes the CleanDataAccuracy metric.
        """
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, clean_mask: torch.Tensor | None) -> None:
        """
        Updates the metric state with predictions, targets, and optional clean mask.

        Args:
            preds (torch.Tensor): The model predictions of shape (batch_size, num_classes).
            targets (torch.Tensor): The ground truth labels of shape (batch_size).
            clean_mask (Optional[torch.Tensor]): A binary mask indicating clean samples. If None, all samples are used.
        """
        if clean_mask is not None:
            preds = preds[clean_mask]

        preds = preds.argmax(dim=-1)
        targets = targets[clean_mask]

        self.correct += (preds == targets).sum()
        self.total += len(preds)

    def compute(self) -> torch.Tensor:
        """
        Computes the accuracy on clean data.

        Returns:
            torch.Tensor: The clean data accuracy (correct/total).
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
    asr.update(preds, targets, poison_mask)
    cda.update(preds, targets, clean_mask)

    # compute results
    print(f"ASR: {asr.compute().item()}")
    print(f"CDA: {cda.compute().item()}")
