import pytorch_lightning as pl
import torch
import torchmetrics
from torch.nn import CrossEntropyLoss

from losses import DiceLoss


class Segmentation_UNET(pl.LightningModule):
    def __init__(self, model, lr, num_classes, weight_ce, weight_dice, metrics=True):
        super().__init__()
        # model
        self.model = model

        # learning rate
        self.lr = lr

        # number of classes
        self.num_classes = num_classes

        # loss
        self.register_buffer("weight_ce", weight_ce)
        self.register_buffer("weight_dice", weight_dice)
        self.dice_loss = DiceLoss(weight=self.weight_dice)
        self.ce_loss = CrossEntropyLoss(weight=self.weight_ce)

        # save hyperparameters
        self.save_hyperparameters()

        # metrics
        self.metrics = metrics
        if self.metrics:
            self.f1_train = CustomMetric(
                metric=torchmetrics.functional.f1,
                metric_name="F1",
                num_classes=self.num_classes,
                average="none",
                mdmc_average="samplewise",
            )
            self.f1_valid = CustomMetric(
                metric=torchmetrics.functional.f1,
                metric_name="F1",
                num_classes=self.num_classes,
                average="none",
                mdmc_average="samplewise",
            )
            self.f1_test = CustomMetric(
                metric=torchmetrics.functional.f1,
                metric_name="F1",
                num_classes=self.num_classes,
                average="none",
                mdmc_average="samplewise",
            )

            self.iou_train = CustomMetric(
                metric=torchmetrics.functional.iou,
                metric_name="IoU",
                num_classes=self.num_classes,
                reduction="none",
            )

            self.iou_valid = CustomMetric(
                metric=torchmetrics.functional.iou,
                metric_name="IoU",
                num_classes=self.num_classes,
                reduction="none",
            )

            self.iou_test = CustomMetric(
                metric=torchmetrics.functional.iou,
                metric_name="IoU",
                num_classes=self.num_classes,
                reduction="none",
            )

    def shared_step(self, batch):
        # Batch
        x, y, x_name, y_name = batch["x"], batch["y"], batch["x_name"], batch["y_name"]

        # Prediction
        out = self.model(x)

        # Softmax
        out_soft = torch.nn.functional.softmax(out, dim=1)

        # Loss
        ce_loss = self.ce_loss(out, y)  # cross entropy loss (LogSoftmax + NLLLoss)
        dice_loss = self.dice_loss(out, y)  # soft dice loss (Softmax + soft dice)
        loss = (ce_loss + dice_loss) / 2  # Linear combination of both losses

        return {**batch, "pred": out_soft, "loss": loss}

    def training_step(self, batch, batch_idx):
        # Loss
        shared_step = self.shared_step(batch)

        # Metrics
        if self.metrics:
            self.compute_and_log_metrics_batch(
                pred=shared_step["pred"],
                tar=shared_step["y"],
                name_phase="Train",
                metrics_module=self.f1_train,
            )  # F1

            self.compute_and_log_metrics_batch(
                pred=torchmetrics.utilities.data.to_categorical(shared_step["pred"]),
                tar=shared_step["y"],
                name_phase="Train",
                metrics_module=self.iou_train,
            )  # IoU

        return shared_step["loss"]

    def training_epoch_end(self, outputs):
        if self.metrics:
            self.compute_and_log_metrics_epoch(
                name_phase="Train", metrics_module=self.f1_train
            )  # F1
            self.compute_and_log_metrics_epoch(
                name_phase="Train", metrics_module=self.iou_train
            )  # IoU

    def validation_step(self, batch, batch_idx):
        # Loss
        shared_step = self.shared_step(batch)

        # Metrics
        if self.metrics:
            self.compute_and_log_metrics_batch(
                pred=shared_step["pred"],
                tar=shared_step["y"],
                name_phase="Valid",
                metrics_module=self.f1_valid,
            )  # F1

            self.compute_and_log_metrics_batch(
                pred=pl.metrics.utils.to_categorical(shared_step["pred"]),
                tar=shared_step["y"],
                name_phase="Valid",
                metrics_module=self.iou_valid,
            )  # IoU

            # Logging for checkpoint
            self.log(
                "checkpoint_valid_f1_epoch", self.f1_valid.get_metrics_batch(mean=True)
            )  # per epoch automatically

    def validation_epoch_end(self, outputs):
        if self.metrics:
            self.compute_and_log_metrics_epoch(
                name_phase="Valid", metrics_module=self.f1_valid
            )  # F1
            self.compute_and_log_metrics_epoch(
                name_phase="Valid", metrics_module=self.iou_valid
            )  # IoU

    def test_step(self, batch, batch_idx):
        # Loss
        shared_step = self.shared_step(batch)

        # Metrics
        if self.metrics:
            self.compute_and_log_metrics_batch(
                pred=shared_step["pred"],
                tar=shared_step["y"],
                name_phase="Test",
                metrics_module=self.f1_test,
                name=shared_step["x_name"],
            )  # F1

            self.compute_and_log_metrics_batch(
                pred=pl.metrics.utils.to_categorical(shared_step["pred"]),
                tar=shared_step["y"],
                name_phase="Test",
                metrics_module=self.iou_test,
                name=shared_step["x_name"],
            )  # IoU

            # Names
            if shared_step["y"].shape[0] == 1:
                # Log the name of target only if batch_size=1
                # Logging a list of strings is not yet supported
                self.log_names_batch(name_phase="Test", name=shared_step["x_name"][0])

    def test_epoch_end(self, outputs):
        if self.metrics:
            self.compute_and_log_metrics_epoch(
                name_phase="Valid", metrics_module=self.f1_test
            )  # F1
            self.compute_and_log_metrics_epoch(
                name_phase="Valid", metrics_module=self.iou_test
            )  # IoU

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.75, patience=10, min_lr=0
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": "checkpoint_valid_f1_epoch",
        }

    def compute_and_log_metrics_batch(
        self, pred, tar, name_phase, metrics_module, name=None
    ):
        # Metrics
        metrics_module.batch(pred, tar, name=name)  # e.g. [0.2, 0.3, 0.25, 0.25]

        # Logging mean
        self.logger.experiment.log_metric(
            f"{name_phase}/{metrics_module}/Batch",
            metrics_module.get_metrics_batch(mean=True),
        )

        # Logging per class
        for class_idx, metric in zip(
            metrics_module.valid_class, metrics_module.get_metrics_batch(mean=False)
        ):
            self.logger.experiment.log_metric(
                f"{name_phase}/{metrics_module}/Batch/Class/{class_idx}", metric
            )

    def compute_and_log_metrics_epoch(self, name_phase, metrics_module):
        # Class
        for class_idx, value in enumerate(metrics_module.get_metrics_epoch()):
            self.logger.experiment.log_metric(
                f"{name_phase}/{metrics_module}/Epoch/Class/{class_idx}", value
            )

        # Total
        self.logger.experiment.log_metric(
            f"{name_phase}/{metrics_module}/Epoch", metrics_module.epoch()
        )  # Total F1

    def log_names_batch(self, name_phase, name):
        self.logger.experiment.log_text(f"{name_phase}/Batch/Names", name)


class CustomMetric:
    def __init__(self, metric, metric_name, **kwargs):
        self.metric = metric
        self.metric_name = metric_name
        self.kwargs = kwargs

        self.scores = []
        self.valid_classes = []
        self.valid_matrices = []
        self.names = []

        self.score = None
        self.valid_class = None
        self.valid_matrix = None
        self.name = None

        self.last_scores = None
        self.last_valid_classes = None
        self.last_valid_matrices = None
        self.last_names = None

    def batch(self, prediction, target, name=None):
        # compute scores for every batch
        self.score = self.metric(prediction, target, **self.kwargs).to("cpu")
        # compute valid classes for every batch
        self.valid_class = target.unique().to("cpu")
        # compute valid_matrix for every batch
        dummy = torch.zeros_like(self.score).to("cpu")
        dummy[self.valid_class] = 1
        self.valid_matrix = dummy.type(torch.bool).to("cpu")

        self.scores.append(self.score)
        self.valid_classes.append(self.valid_class)
        self.valid_matrices.append(self.valid_matrix)

        # store name(s)
        if name:
            self.name = name
            self.names.append(self.name)

    def get_metrics_batch(self, mean=True):
        # returns the class metrics of the batch for the classes that are present in the image
        if mean:
            return self.score[self.valid_class].mean()
        else:
            return self.score[self.valid_class]

    def get_metrics_epoch(self, last=False, transpose=True):
        # transpose=True gives the per class metrics (mean)
        # transpose=False, gives the per batch metrics (mean)
        if last:
            if transpose:
                scores = torch.stack(self.last_scores).T
                masks = torch.stack(self.last_valid_matrices).T
            else:
                scores = torch.stack(self.last_scores)
                masks = torch.stack(self.last_valid_matrices)
        else:
            if transpose:
                scores = torch.stack(self.scores).T
                masks = torch.stack(self.valid_matrices).T
            else:
                scores = torch.stack(self.scores)
                masks = torch.stack(self.valid_matrices)

        # iterate over columns (classes) and only select the present classes
        filtered = [s[m] for s, m in zip(scores, masks)]

        return torch.stack([c.mean() for c in filtered])

    def epoch(self):
        # compute scores for every epoch

        self.last_scores = self.scores
        self.last_valid_classes = self.valid_classes
        self.last_valid_matrices = self.valid_matrices
        self.last_names = self.names

        result = self.get_metrics_epoch()

        self.reset()
        return result.mean()

    def reset(self):
        self.scores = []
        self.valid_classes = []
        self.valid_matrices = []
        self.names = []

    def __repr__(self):
        return self.metric_name
