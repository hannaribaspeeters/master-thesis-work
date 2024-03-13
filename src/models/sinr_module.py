from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassF1Score,
    MultilabelAUROC,
    MultilabelF1Score,
)

from src.criterions import abstract_criterion
from src.utils.adapted_torchm_auroc import AdaptedMultilabelAUROC


class SINRModule(LightningModule):
    """SINR implementation using a combinations of all 1-D inputs passed to it."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: abstract_criterion,
        compile: bool,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["net"], logger=False)

        self.net = net

        # loss function
        self.criterion = criterion

        # metric objects for calculating and averaging across batches
        self.train_f1 = MultilabelF1Score(num_labels=10040, average="micro")
        self.val_f1 = MultilabelF1Score(num_labels=10040, average="micro")
        self.test_f1 = MultilabelF1Score(num_labels=10040, average="micro")

        # Had to write an adapted class making use of the weighted functionality to make the metric ignore classes without support in val/test
        # This was more lightweight than own reimplementation, although now the average="weighted" is slightly unintuitive
        self.val_auroc_macro = AdaptedMultilabelAUROC(num_labels=10040, average="weighted")
        self.test_auroc_macro = AdaptedMultilabelAUROC(num_labels=10040, average="weighted")

        self.val_auroc_weighted = MultilabelAUROC(num_labels=10040, average="weighted")
        self.test_auroc_weighted = MultilabelAUROC(num_labels=10040, average="weighted")

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_f1_best = MaxMetric()
        self.val_auroc_macro_best = MaxMetric()
        self.val_auroc_weighted_best = MaxMetric()

    def forward(self, batch) -> torch.Tensor:
        """Forwarding batch through the network.

        :param batch: Dictionary containing a variety of predictors.
        :return: a tensor of predictions
        """

        x = []
        # We are simply appending all 1-D predictors in the batch_dict
        # The config is responsible to make sure that input_dim and predictors are set accordingly

        for key, value in batch.items():
            if key != "y" and len(value.shape) == 2:  # True if predictor is 1-D
                x.append(value)
        x = torch.concat(x, dim=1)

        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_f1_best.reset()
        self.val_auroc_macro_best.reset()
        self.val_auroc_weighted_best.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: x.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """
        logits = self.forward(batch)
        y = batch["y"]
        loss = self.criterion(logits, y)
        y = y.type(torch.int)
        return loss, logits, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: x.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_f1(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True, prog_bar=True)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: x.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_f1(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.val_auroc_macro(preds, targets)
        self.val_auroc_weighted(preds, targets)
        self.log(
            "val/auroc_macro", self.val_auroc_macro, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "val/auroc_weighted",
            self.val_auroc_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        f1 = self.val_f1.compute()  # get current val f1
        self.val_f1_best(f1)  # update best so far val f1
        self.log("val/f1_best", self.val_f1_best.compute(), sync_dist=True, prog_bar=True)
        auroc_macro = self.val_auroc_macro.compute()
        self.val_auroc_macro_best(auroc_macro)
        self.log(
            "val/auroc_macro_best",
            self.val_auroc_macro_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )
        auroc_weighted = self.val_auroc_weighted.compute()
        self.val_auroc_weighted_best(auroc_weighted)
        self.log(
            "val/auroc_weighted_best",
            self.val_auroc_weighted_best.compute(),
            sync_dist=True,
            prog_bar=True,
        )

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_f1(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True, prog_bar=True)

        self.test_auroc_macro(preds, targets)
        self.test_auroc_weighted(preds, targets)
        self.log(
            "test/auroc_macro", self.test_auroc_macro, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "test/auroc_weighted",
            self.test_auroc_weighted,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    _ = SINRModule(None, None, None, None, None)
