from abc import ABC


class AbstractCriterion(ABC):
    def __init__(self):
        pass

    def __call__(self, logits, y):
        raise NotImplementedError()

    def set_model(self, model):
        self.model = model

    def set_dataset(self, dataset):
        self.dataset = dataset
