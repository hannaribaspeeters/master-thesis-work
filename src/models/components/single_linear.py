import torch
from torch import nn


class SingleLinear(nn.Module):
    """A simple fully-connected neural net for computing predictions."""

    def __init__(
        self,
        input_size: int = 4,
        output_size: int = 10040,
    ) -> None:
        """Initialize a `SingleLinear` module.

        :param input_size: The number of input features.
        :param output_size: The number of output features of the linear layer.
        """
        super().__init__()

        self.model = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.model(x)


if __name__ == "__main__":
    _ = SingleLinear()
