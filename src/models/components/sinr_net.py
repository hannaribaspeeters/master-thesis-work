import torch
from torch import nn


class SinrResidualBlock(nn.Module):
    """Residal block for SINR."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        """Initialize a `SinrResidualBlock` module.

        :param hidden_dim: The number of hidden_features.
        :param dropout: The amount of dropout applied.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.layers(x) + x


class SinrNet(nn.Module):
    """The SINR net consisting of residual blocks."""

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 512,
        num_blocks: int = 8,
        output_classes: int = 10040,
        dropout: float = 0.3,
    ) -> None:
        """Initialize a `SinrNet` module.

        :param input_dim: The number of input features.
        :param hidden_dim: The input/output size of the hidden layers.
        :param num_layers: The number of residual blocks.
        :param output_classes: The number of output classes of the network.
        :param dropout: The amount of dropout applied.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            *[SinrResidualBlock(hidden_dim, dropout) for i in range(num_blocks)]
        )

        self.classifier = nn.Linear(hidden_dim, output_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        x = self.net(x)
        return self.classifier(x)


if __name__ == "__main__":
    _ = SinrNet()
