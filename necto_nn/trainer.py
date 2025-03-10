import numpy as np
from tqdm import tqdm
from itertools import batched, chain, repeat
from .losses import Loss
from .optimizers import Optimizer
from .metrics import get_metric
from .nn import FeedForwardNetwork
import wandb


def array_batched(array, n_epochs, batch_size):
    return map(
        np.array,
        batched(
            chain.from_iterable(repeat(array.tolist(), n_epochs)),
            batch_size,
        ),
    )


class Trainer:
    def __init__(
        self,
        model: FeedForwardNetwork,
        n_epochs,
        batch_size,
        optimizer: Optimizer,
        loss: Loss,
        metrics=None,
        eval_steps=None,
        wandb_log=False,
    ):
        self.model = model
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics
        self.train_metrics = {
            metric for metric in metrics if not metric.startswith("val_")
        }
        self.val_metrics = {metric[4:] for metric in metrics if metric.startswith("val_")}
        self.eval_steps = eval_steps
        self.wandb_log = wandb_log

    def eval(self, X, y, metrics):
        y_pred = self.model.forward(X)
        loss = self.loss(y_pred, y)
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)
        return {"loss": loss} | {
            metric: get_metric(metric)(y, y_pred) for metric in metrics
        }

    def train(
        self,
        X_train,
        y_train,
        X_val=None,
        y_val=None,
    ):
        data_size = X_train.shape[0]
        total_size = data_size * self.n_epochs

        n_steps = (total_size // self.batch_size) + bool(total_size % self.batch_size)
        p_bar = tqdm(
            zip(
                array_batched(X_train, self.n_epochs, self.batch_size),
                array_batched(y_train, self.n_epochs, self.batch_size),
            ),
            total=n_steps,
            desc="steps",
        )
        self.model.init()
        self.optimizer.init(
            self.model,
        )
        completed = 0
        p_bar.set_postfix({"epoch": 0})
        val_metrics = {}
        for step, (X_batch, y_batch) in enumerate(p_bar, 1):
            y_pred = self.model.forward(X_batch)
            loss = self.loss(y_batch, y_pred)
            loss_gradient = self.loss.gradient(y_batch, y_pred)
            self.model.backward(loss_gradient)
            self.optimizer.update()
            completed += X_batch.shape[0]

            train_metrics = {
                "epoch": completed / data_size,
                "loss": loss,
            } | {
                metric: get_metric(metric)(
                    y_batch.argmax(axis=-1), y_pred.argmax(axis=-1)
                )
                for metric in self.train_metrics
            }

            if self.eval_steps and step % self.eval_steps == 0:
                val_metrics = {
                    f"val_{k}": v
                    for k, v in self.eval(X_val, y_val, self.val_metrics).items()
                }

                if self.wandb_log:
                    wandb.log(data=val_metrics, step=step)
            if self.wandb_log:
                wandb.log(
                    data=train_metrics,
                    step=step,
                )
            p_bar.set_postfix(
                {
                    **train_metrics,
                    **val_metrics,
                }
            )
