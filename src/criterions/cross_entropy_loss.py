import torch

class CEL():
    def __init__(self, model = None, dataset = None):
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, logits, y):
        return self.criterion(logits, y)