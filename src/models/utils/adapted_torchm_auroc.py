from typing import Any, List, Literal, Optional, Sequence, Tuple, Type, Union

import torch
from torch import Tensor
from torchmetrics.classification import MultilabelAUROC
from torchmetrics.functional.classification.auroc import (
    _binary_auroc_compute,
    _reduce_auroc,
)
from torchmetrics.functional.classification.roc import _multilabel_roc_compute
from torchmetrics.utilities.data import dim_zero_cat


def _multilabel_auroc_compute_adapted(
    state: Union[Tensor, Tuple[Tensor, Tensor]],
    num_labels: int,
    average: Optional[Literal["micro", "macro", "weighted", "none"]],
    thresholds: Optional[Tensor],
    ignore_index: Optional[int] = None,
) -> Tensor:
    """Want to change the weights to not take the absolute value when counting the classes, but
    rather binary presence.

    This way, only classes that are present get a weight 1, all others 0.
    """
    if average == "micro":
        if isinstance(state, Tensor) and thresholds is not None:
            return _binary_auroc_compute(state.sum(1), thresholds, max_fpr=None)

        preds = state[0].flatten()
        target = state[1].flatten()
        if ignore_index is not None:
            idx = target == ignore_index
            preds = preds[~idx]
            target = target[~idx]
        return _binary_auroc_compute((preds, target), thresholds, max_fpr=None)

    fprate, tprate, _ = _multilabel_roc_compute(state, num_labels, thresholds, ignore_index)

    # Adaption: We are effectively capping the weights at 1
    weights = (state[1] == 1).sum(dim=0).float()
    ones = Tensor([1]).to(weights.device).expand_as(weights)
    weights = torch.minimum(weights, ones).to(weights.device)

    return _reduce_auroc(
        fprate,
        tprate,
        average,
        # Original: weights=(state[1] == 1).sum(dim=0).float() if thresholds is None else state[0][:, 1, :].sum(-1)
        # Adapted:
        weights=weights if thresholds is None else state[0][:, 1, :].sum(-1),
    )


class AdaptedMultilabelAUROC(MultilabelAUROC):
    """Extending the MultilabelAUROC to be able to ignore classes that are not present in the test
    data."""

    def __init__(
        self,
        num_labels: int,
        average: Optional[Literal["micro", "macro", "weighted", "none"]] = "macro",
        thresholds: Optional[Union[int, List[float], Tensor]] = None,
        ignore_index: Optional[int] = None,
        validate_args: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(num_labels, average, thresholds, ignore_index, validate_args, **kwargs)

        if average != "weighted" or thresholds is not None:
            raise ValueError(
                "Only use this class with weighted (which is actually macro) without thresholds."
            )

    def compute(self) -> Tensor:  # type: ignore[override]
        """Compute metric."""
        state = (dim_zero_cat(self.preds), dim_zero_cat(self.target))
        return _multilabel_auroc_compute_adapted(
            state, self.num_labels, self.average, self.thresholds, self.ignore_index
        )
