import math

import pandas as pd
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm.notebook import tqdm, trange


class LearningRateFinder:
    """
    Train a model using different learning rates within a range to find the optimal learning rate.
    """

    def __init__(self, model: nn.Module, criterion, optimizer, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.loss_history = {}
        self._model_init = model.state_dict()
        self._opt_init = optimizer.state_dict()
        self.device = device

    def fit(
        self,
        data_loader: DataLoader,
        steps: int = 100,
        min_lr: float = 1e-7,
        max_lr: float = 1,
        constant_increment: bool = False,
    ):
        """
        Trains the model for number of steps using varied learning rate and store the statistics
        """
        self.loss_history = {}
        self.model.train()
        current_lr = min_lr
        steps_counter = 0
        epochs = math.ceil(steps / len(data_loader))

        progressbar = trange(epochs, desc="Progress")
        for epoch in progressbar:
            batch_iter = tqdm(
                enumerate(data_loader), "Training", total=len(data_loader), leave=False
            )

            for i, (x, y) in batch_iter:
                x, y = x.to(self.device), y.to(self.device)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = current_lr
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y)
                loss.backward()
                self.optimizer.step()
                self.loss_history[current_lr] = loss.item()

                steps_counter += 1
                if steps_counter > steps:
                    break

                if constant_increment:
                    current_lr += (max_lr - min_lr) / steps
                else:
                    current_lr = current_lr * (max_lr / min_lr) ** (1 / steps)

    def plot(
        self,
        smoothing: bool = True,
        clipping: bool = True,
        smoothing_factor: float = 0.1,
    ):
        """
        Shows loss vs learning rate(log scale) in a matplotlib plot
        """
        loss_data = pd.Series(list(self.loss_history.values()))
        lr_list = list(self.loss_history.keys())
        if smoothing:
            loss_data = loss_data.ewm(alpha=smoothing_factor).mean()
            loss_data = loss_data.divide(
                pd.Series(
                    [
                        1 - (1.0 - smoothing_factor) ** i
                        for i in range(1, loss_data.shape[0] + 1)
                    ]
                )
            )  # bias correction
        if clipping:
            loss_data = loss_data[10:-5]
            lr_list = lr_list[10:-5]
        plt.plot(lr_list, loss_data)
        plt.xscale("log")
        plt.title("Loss vs Learning rate")
        plt.xlabel("Learning rate (log scale)")
        plt.ylabel("Loss (exponential moving average)")
        plt.show()

    def reset(self):
        """
        Resets the model and optimizer to its initial state
        """
        self.model.load_state_dict(self._model_init)
        self.optimizer.load_state_dict(self._opt_init)
        print("Model and optimizer in initial state.")
