import torch

from src.criterions.abstract_criterion import AbstractCriterion


class CEL(AbstractCriterion):
    def __init__(self):
        super().__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, y):
        return self.criterion(logits, y)
