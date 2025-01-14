import torch
from torchmetrics import Metric


class AttackSuccessRate(Metric):
    def __init__(self, target_index: int | None = None):
        super().__init__()
        self.target_class = target_index
        self.add_state("success", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, poison_mask: torch.Tensor | None) -> None:
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
        return self.success.float() / self.total


class CleanDataAccuracy(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor, clean_mask: torch.Tensor | None) -> None:
        if clean_mask is not None:
            preds = preds[clean_mask]

        preds = preds.argmax(dim=-1)
        targets = targets[clean_mask]

        self.correct += (preds == targets).sum()
        self.total += len(preds)

    def compute(self) -> torch.Tensor:
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
