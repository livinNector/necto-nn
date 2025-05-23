from copy import deepcopy

import numpy as np
from numba import typed
from tqdm import tqdm

import wandb

from .losses import Loss
from .metrics import get_metric
from .nn import FeedForwardNetwork
from .optimizers import Optimizer


def array_batched(a, batch_size, epochs):
    n = a.shape[0]
    m = n * epochs
    remaining = m % batch_size
    m = m // batch_size + bool(remaining)
    if remaining:
        a = np.vstack([a, a[: batch_size - remaining]])
    for step in range(m):
        start = (step * batch_size) % n
        yield a[start : start + batch_size]


def array_batch_starts(a, batch_size, epochs):
    n = a.shape[0]
    m = n * epochs
    m = m // batch_size + bool(m % batch_size)
    return np.array([(step * batch_size) % n for step in range(m)])


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
        metric_for_best_model="val_accuracy",
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
        self.val_metrics = {
            metric[4:] for metric in metrics if metric.startswith("val_")
        }
        self.eval_steps = eval_steps
        self.metric_for_best_model = metric_for_best_model
        self.wandb_log = wandb_log

    def eval(self, X, y, metrics):
        y_pred = self.model.forward(X)
        loss = self.loss.forward(y_pred, y)
        y_pred = np.argmax(y_pred, axis=-1)
        y = np.argmax(y, axis=-1)
        return {"loss": loss} | {
            metric: get_metric(metric)(y, y_pred) for metric in metrics
        }

    def compute_metrics(self, X_train, y_train, X_val, y_val):
        return {
            k: v for k, v in self.eval(X_train, y_train, self.train_metrics).items()
        } | {
            f"val_{k}": v for k, v in self.eval(X_val, y_val, self.val_metrics).items()
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
        X_train, X_val = X_train.astype(float), X_val.astype(float)

        n_steps = (total_size // self.batch_size) + bool(total_size % self.batch_size)
        p_bar = tqdm(
            zip(
                array_batched(
                    X_train,
                    self.batch_size,
                    self.n_epochs,
                ),
                array_batched(
                    y_train,
                    self.batch_size,
                    self.n_epochs,
                ),
            ),
            total=n_steps,
            desc="steps",
            mininterval=0.5,
        )
        compiled_model = self.model.get_compiled()
        self.optimizer.init(
            typed.List(
                list(compiled_model.weights)
                + [x[np.newaxis, :] for x in compiled_model.biases]
            ),
            typed.List(
                list(compiled_model.d_weights)
                + [x[np.newaxis, :] for x in compiled_model.d_biases]
            ),
        )
        p_bar.set_postfix({"epoch": 0})
        metric_best = None
        for step, (X_batch, y_batch) in enumerate(p_bar, 1):
            y_pred = compiled_model.forward(X_batch)
            loss_gradient = self.loss.gradient(y_batch, y_pred)
            compiled_model.backward(loss_gradient)
            self.optimizer.update()
            if self.eval_steps and (step % self.eval_steps == 0 or step == n_steps):
                metrics = self.compute_metrics(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                )
                metrics["epoch"] = self.n_epochs * step / n_steps
                if self.metric_for_best_model:
                    metric_curr = metrics[self.metric_for_best_model]
                    if metric_best is None or metric_curr > metric_best:
                        metric_best = metric_curr
                        best_params = deepcopy(
                            list(self.model.weights) + list(self.model.biases)
                        )
                p_bar.set_postfix({k: f"{v:.3f}" for k, v in metrics.items()})
                if self.wandb_log:
                    wandb.log(
                        data=metrics,
                        step=step,
                    )
        # loading the best params at the end
        self.model.weights = typed.List(best_params[: self.model.n_layers])
        self.model.biases = typed.List(best_params[self.model.n_layers :])
