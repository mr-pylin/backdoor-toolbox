import torch
from torchmetrics import Metric


class CleanDataAccuracy(Metric):
    """
    Metric to calculate the accuracy on clean data.

    Attributes:
        correct (torch.Tensor): Counter for the number of correct predictions.
        total (torch.Tensor): Counter for the total number of samples.

    Methods:
        update(preds, targets):
            Updates the counters with predictions and targets from the current batch.
        compute():
            Computes the accuracy as the ratio of correct predictions to total samples.
    """

    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, clean_mask: torch.Tensor) -> None:
        """
        Updates the metric state with predictions and targets from the current batch.

        Args:
            preds (torch.Tensor): Predictions from the model, assumed to be logits.
            targets (torch.Tensor): Ground truth labels.
            clean_mask (torch.Tensor): Boolean mask for clean samples.
        """

        clean_preds = preds[clean_mask].argmax(dim=-1)
        clean_targets = targets[clean_mask]
        self.correct += (clean_preds == clean_targets).sum()
        self.total += clean_mask.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the final accuracy based on accumulated state.

        Returns:
            torch.Tensor: The accuracy as a scalar tensor.
        """

        return self.correct.float() / self.total


class AttackSuccessRate(Metric):
    """
    Metric to calculate the attack success rate (ASR) for backdoor attacks.

    Attributes:
        target_index (int): The target class for the backdoor attack.
        success (torch.Tensor): Counter for the number of successful attacks.
        total (torch.Tensor): Counter for the total number of poisoned samples.

    Methods:
        update(preds, poisoned_targets):
            Updates the counters with predictions and poisoned targets from the current batch.
        compute():
            Computes the ASR as the ratio of successful attacks to total poisoned samples.
    """

    def __init__(self, target_index: int | None = None):
        super().__init__()
        self.target_class = target_index
        self.add_state("success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, poison_mask: torch.Tensor) -> None:
        """
        Updates the metric state with predictions and poisoned targets from the current batch.

        Args:
            preds (torch.Tensor): Predictions from the model, assumed to be logits.
            poisoned_targets (torch.Tensor): Ground truth labels for poisoned samples.
        """

        preds = preds[poison_mask].argmax(dim=-1)
        if self.target_class is not None:
            self.success += (preds == self.target_class).sum()
        else:
            targets = targets[poison_mask]
            self.success += (preds == targets).sum()
        self.total += poison_mask.sum()

    def compute(self) -> torch.Tensor:
        """
        Computes the final attack success rate based on accumulated state.

        Returns:
            torch.Tensor: The ASR as a scalar tensor.
        """

        return self.success.float() / self.total


if __name__ == "__main__":

    # simulated data
    preds = torch.tensor([[0.1, 0.9], [0.8, 0.2], [0.7, 0.3]])
    targets = torch.tensor([1, 0, 1])
    clean_mask = torch.tensor([True, False, True])
    poison_mask = ~clean_mask

    # instantiate metrics
    cda = CleanDataAccuracy()
    asr = AttackSuccessRate(target_index=0)

    # update metrics
    cda.update(preds, targets, clean_mask)
    asr.update(preds, targets, poison_mask)

    # compute results
    print(f"CDA: {cda.compute().item()}")
    print(f"ASR: {asr.compute().item()}")
