from abc import ABC


class AbstractCriterion(ABC):
    """Abstract class for all criterions."""

    def __init__(self):
        pass

    def __call__(self, logits, y):
        """Gets called by the model.

        :params logits: Output of the network without softmax/sigmoid.
        :params y: binary torch.Tensor of size [batch_size, num_classes].
        """
        raise NotImplementedError()

    def set_model(self, model):
        """Abstract function to let loss have access to the model for background sampling."""
        pass

    def set_dataset(self, dataset):
        """Abstract function to let loss have access to the dataset/predictors for background
        sampling."""
        pass
