import argparse
import wandb
from keras._tf_keras.keras.datasets import fashion_mnist, mnist
from necto_nn.nn import FeedForwardNetwork
from necto_nn.activations import get_activation
from necto_nn.optimizers import get_optimizer
from necto_nn.losses import get_loss
from necto_nn.intializers import get_initializer
from necto_nn.trainer import Trainer
from necto_nn.utils import one_hot, train_test_split, flatten


datasets = {"fashion_mnist": fashion_mnist, "mnist": mnist}


wandb.login()


def train(
    wandb_project: str,
    wandb_entity: str,
    dataset: str,
    epochs: int,
    batch_size: int,
    loss: str,
    optimizer: str,
    learning_rate: float,
    momentum: float,
    beta: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
    weight_init: str,
    num_layers: int,
    hidden_size: int,
    activation: str,
):
    """
    Train a neural network with specified parameters.
    """

    (X_train, y_train), (X_test, y_test) = datasets[dataset].load_data()
    X_train, X_test = flatten(X_train), flatten(X_test)
    X_train, X_test = flatten(X_train), flatten(X_test)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1)
    y_train, y_val, y_test = one_hot(y_train), one_hot(y_val), one_hot(y_test)
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]

    activation_func = get_activation(activation)
    model = FeedForwardNetwork(
        input_size=input_size,
        layers=[
            *((hidden_size, activation_func()) for i in range(num_layers)),
            (output_size, get_activation("softmax")()),
        ],
        initializer=get_initializer(weight_init)(),
        weight_decay=weight_decay,
    )

    loss_func = get_loss(loss)

    match optimizer:
        case "sgd":
            optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
            )
        case "momentum":
            optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
                momentum=momentum,
            )
        case "nag":
            optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
                momentum=momentum,
            )
        case "rmsprop":
            optimizer = get_optimizer(
                optimizer, learning_rate=learning_rate, beta=beta, eps=epsilon
            )
        case "adam":
            optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                eps=epsilon,
            )
        case "nadam":
            optimizer = get_optimizer(
                optimizer,
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                eps=epsilon,
            )
    if wandb_project:
        wandb.init(project=wandb_project, entity=wandb_entity)
        wandb.define_metric("accuracy", summary="max")
        wandb.define_metric("loss", summary="min")
        wandb.define_metric("val_accuracy", summary="max")
        wandb.define_metric("val_loss", summary="min")
        wandb.define_metric("val_f1_score", summary="max")

    trainer = Trainer(
        model=model,
        n_epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        loss=loss_func,
        metrics=["accuracy", "val_accuracy", "val_f1_score"],
        eval_steps=100,
        wandb_log=bool(wandb_project),
    )
    trainer.train(X_train, y_train, X_val, y_val)
    metrics = trainer.eval(X_test, y_test, metrics=["accuracy", "f1_score"])
    
    if wandb_project:
        for k, v in metrics.items():
            wandb.run.summary[f"test_{k}"] = v
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network with specified parameters."
    )
    parser.add_argument(
        "-wp",
        "--wandb_project",
        type=str,
        default=None,
        help="Project name used to track experiments in Weights & Biases dashboard",
    )
    parser.add_argument(
        "-we",
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb Entity used to track experiments in the Weights & Biases dashboard",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        choices=["mnist", "fashion_mnist"],
        default="fashion_mnist",
        help="Dataset selection",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="Number of epochs to train neural network",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=4,
        help="Batch size used to train neural network",
    )
    parser.add_argument(
        "-l",
        "--loss",
        choices=["mean_squared_error", "cross_entropy"],
        default="cross_entropy",
        help="Loss function",
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
        default="sgd",
        help="Optimizer selection",
    )
    parser.add_argument(
        "-lr",
        "--learning_rate",
        type=float,
        default=0.1,
        help="Learning rate used to optimize model parameters",
    )
    parser.add_argument(
        "-m",
        "--momentum",
        type=float,
        default=0.5,
        help="Momentum used by momentum and nag optimizers",
    )
    parser.add_argument(
        "-beta",
        "--beta",
        type=float,
        default=0.5,
        help="Beta used by rmsprop optimizer",
    )
    parser.add_argument(
        "-beta1",
        "--beta1",
        type=float,
        default=0.5,
        help="Beta1 used by adam and nadam optimizers",
    )
    parser.add_argument(
        "-beta2",
        "--beta2",
        type=float,
        default=0.5,
        help="Beta2 used by adam and nadam optimizers",
    )
    parser.add_argument(
        "-eps",
        "--epsilon",
        type=float,
        default=0.000001,
        help="Epsilon used by optimizers",
    )
    parser.add_argument(
        "-w_d",
        "--weight_decay",
        type=float,
        default=0.0,
        help="Weight decay used by optimizers",
    )
    parser.add_argument(
        "-w_i",
        "--weight_init",
        choices=["random", "xavier"],
        default="random",
        help="Weight initialization method",
    )
    parser.add_argument(
        "-nhl",
        "--num_layers",
        type=int,
        default=1,
        help="Number of hidden layers used in feedforward neural network",
    )
    parser.add_argument(
        "-sz",
        "--hidden_size",
        type=int,
        default=4,
        help="Number of hidden neurons in a feedforward layer",
    )
    parser.add_argument(
        "-a",
        "--activation",
        choices=["identity", "sigmoid", "tanh", "ReLU"],
        default="sigmoid",
        help="Activation function",
    )

    args = parser.parse_args()
    train(
        args.wandb_project,
        args.wandb_entity,
        args.dataset,
        args.epochs,
        args.batch_size,
        args.loss,
        args.optimizer,
        args.learning_rate,
        args.momentum,
        args.beta,
        args.beta1,
        args.beta2,
        args.epsilon,
        args.weight_decay,
        args.weight_init,
        args.num_layers,
        args.hidden_size,
        args.activation,
    )
