import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE, CrossEntropy
from src.activations import ReLU, Sigmoid, Tanh, Linear
from src.methods.kmeans import KMeans
from src.utils import normalize_fn, append_bias_term, accuracy_fn, macrof1_fn, mse_fn, label_to_onehot
import os

np.random.seed(100)


def main(args):
    """
    The main function of the script.

    Arguments:
        args (Namespace): arguments that were parsed from the command line (see at the end
                          of this file). Their value can be accessed as "args.argument".
    """


    dataset_path = args.data_path
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")

    ## 1. We first load the data.

    feature_data = np.load(dataset_path, allow_pickle=True)
    train_features, test_features, train_labels_reg, test_labels_reg, train_labels_classif, test_labels_classif = (
        feature_data['xtrain'],feature_data['xtest'],feature_data['ytrainreg'],
        feature_data['ytestreg'],feature_data['ytrainclassif'],feature_data['ytestclassif']
    )

    ## 2. Then we must prepare it. This is where you can create a validation set,
    #  normalize, add bias, etc.

    # Create a validation set unless we want to evaluate on the held-out test split.
    if not args.test:
        split_idx = int(0.85 * train_features.shape[0])

        x_val = train_features[split_idx:]
        y_val_classif = train_labels_classif[split_idx:]
        y_val_reg = train_labels_reg[split_idx:]

        train_features = train_features[:split_idx]
        train_labels_classif = train_labels_classif[:split_idx]
        train_labels_reg = train_labels_reg[:split_idx]

        test_features = x_val
        test_labels_classif = y_val_classif
        test_labels_reg = y_val_reg

    means = np.mean(train_features, axis=0, keepdims=True)
    stds = np.std(train_features, axis=0, keepdims=True)
    stds = np.where(stds == 0, 1, stds)

    train_features = normalize_fn(train_features, means, stds)
    test_features = normalize_fn(test_features, means, stds)

    if args.plot_mlp:
        values = _parse_sweep_values(args.plot_values)
        task_to_use = args.plot_task if args.plot_task is not None else args.task
        plot_mlp_hyperparams(
            train_features,
            train_labels_classif if task_to_use == "classification" else train_labels_reg,
            test_features,
            test_labels_classif if task_to_use == "classification" else test_labels_reg,
            hyperparam=args.plot_hyperparam,
            values=values,
            task=task_to_use,
            hidden_layers=args.hidden_layers,
            hidden_units=args.hidden_units,
            activation=args.activation,
            loss=args.loss,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            filename_prefix=args.plot_filename_prefix,
        )
        return

    if args.plot_kmeans:
        K_values = _parse_sweep_values(args.plot_k_values)
        plot_kmeans_hyperparams(
            train_features,
            train_labels_classif,
            test_features,
            test_labels_classif,
            K_values=K_values,
            max_iters=args.plot_max_iters,
            filename_prefix=args.plot_filename_prefix,
        )
        return

    ## 3. Initialize the method you want to use.

    # Follow the "DummyClassifier" example for your methods
    if args.method == "dummy_classifier":
        method_obj = DummyClassifier(arg1=1, arg2=2)

    elif args.method == "kmeans":
        method_obj = KMeans(K=args.K, max_iters=args.max_iters)

    elif args.method == "mlp":
        activation_map = {
            "relu": ReLU,
            "sigmoid": Sigmoid,
            "tanh": Tanh,
            "linear": Linear,
        }
        hidden_activation = activation_map.get(args.activation.lower())
        if hidden_activation is None:
            raise ValueError(
                f"Unknown activation '{args.activation}'. Choose from: {', '.join(activation_map)}"
            )

        hidden_units = tuple([args.hidden_units] * args.hidden_layers)
        activation_list = tuple([hidden_activation] * args.hidden_layers) + (Linear,)

        if args.task == "classification":
            n_classes = int(np.max(train_labels_classif) + 1)
            method_obj = MLP(
                dimensions=(train_features.shape[1],) + hidden_units + (n_classes,),
                activations=activation_list,
            )
        elif args.task == "regression":
            method_obj = MLP(
                dimensions=(train_features.shape[1],) + hidden_units + (1,),
                activations=activation_list,
            )
        else:
            raise ValueError(f"MLP does not support task: {args.task}")
    else:
        raise ValueError(f"Unknown method: {args.method}")

    ## 4. Train and evaluate the method

    if args.task == "classification":
        if args.method == "mlp":
            train_targets = label_to_onehot(train_labels_classif)
            loss_fn = MSE if args.loss == "mse" else CrossEntropy
            train_predictions = method_obj.fit(
                train_features,
                train_targets,
                loss=loss_fn,
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.lr,
            )
            train_predictions = np.argmax(train_predictions, axis=1)
        else:
            train_predictions = method_obj.fit(train_features, train_labels_classif)

        print(f"Train accuracy: {accuracy_fn(train_predictions, train_labels_classif):.2f}%")
        print(f"Train macro F1: {macrof1_fn(train_predictions, train_labels_classif):.2f}")

        test_predictions = method_obj.predict(test_features)
        if isinstance(test_predictions, np.ndarray) and test_predictions.ndim > 1:
            test_predictions = np.argmax(test_predictions, axis=1)

        print(f"Test accuracy:  {accuracy_fn(test_predictions, test_labels_classif):.2f}%")
        print(f"Test macro F1:  {macrof1_fn(test_predictions, test_labels_classif):.2f}")

    elif args.task == "regression":
        if args.method != "mlp":
            raise ValueError("Regression is only supported with --method mlp for this project setup")
        if args.loss != "mse":
            raise ValueError("Regression only supports MSE loss")

        train_targets = train_labels_reg.reshape(-1, 1)
        pred_values = method_obj.fit(
            train_features,
            train_targets,
            loss=MSE,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
        )
        pred_values = np.ravel(pred_values)
        print(f"Train MSE: {mse_fn(pred_values, train_labels_reg):.6f}")

        pred_values_test = method_obj.predict(test_features)
        pred_values_test = np.ravel(pred_values_test)
        print(f"Test MSE:  {mse_fn(pred_values_test, test_labels_reg):.6f}")

    ### WRITE YOUR CODE HERE if you want to add other outputs, visualization, etc.


def _plot_curve(values, series, xlabel, ylabel, title, filename=None):
    """Plot one or more metric series against a hyperparameter sweep."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to plot hyperparameter curves. "
            "Install it with `pip install matplotlib`."
        ) from exc

    plt.figure(figsize=(8, 5))
    for series_values, label in series:
        plt.plot(values, series_values, marker="o", label=label)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    if any(label is not None for _, label in series):
        plt.legend()
    if filename is not None:
        plt.savefig(filename, bbox_inches="tight")
    plt.show()


def _parse_sweep_values(value_string):
    values = [v.strip() for v in value_string.split(",") if v.strip()]
    parsed = []
    for value in values:
        lowered = value.lower()
        if lowered in {"relu", "sigmoid", "tanh", "linear", "mse", "crossentropy"}:
            parsed.append(lowered)
            continue
        try:
            if "." in value:
                parsed.append(float(value))
            else:
                parsed.append(int(value))
        except ValueError:
            parsed.append(value)
    return parsed


def _build_mlp_model(input_dim, num_classes, hidden_layers, hidden_units, activation):
    activation_map = {
        "relu": ReLU,
        "sigmoid": Sigmoid,
        "tanh": Tanh,
        "linear": Linear,
    }
    hidden_activation = activation_map.get(str(activation).lower())
    if hidden_activation is None:
        raise ValueError(
            f"Unknown hidden activation '{activation}'. "
            f"Choose from: {', '.join(activation_map)}"
        )
    activations = tuple([hidden_activation] * hidden_layers) + (Linear,)
    return MLP(
        dimensions=(input_dim,) + tuple([hidden_units] * hidden_layers) + (num_classes,),
        activations=activations,
    )


def evaluate_mlp_hyperparams(
    train_features,
    train_labels,
    test_features,
    test_labels,
    hyperparam,
    values,
    task,
    hidden_layers,
    hidden_units,
    activation,
    loss,
    epochs,
    batch_size,
    learning_rate,
):
    """Evaluate MLP performance across one hyperparameter sweep.

    For classification, this returns accuracy and macro F1.
    For regression, this returns train/test MSE.
    """
    valid_params = {"hidden_layers", "hidden_units", "activation", "loss", "learning_rate", "batch_size"}
    if hyperparam not in valid_params:
        raise ValueError(f"Unsupported hyperparameter '{hyperparam}'. Choose from {sorted(valid_params)}")
    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'")

    if task == "classification":
        num_classes = int(np.max(train_labels) + 1)
        train_targets = label_to_onehot(train_labels)
    else:
        num_classes = 1
        train_targets = train_labels.reshape(-1, 1)

    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []
    train_mse = []
    test_mse = []

    for value in values:
        value = int(value) if hyperparam in {"hidden_layers", "hidden_units", "batch_size"} else value
        config_hidden_layers = hidden_layers
        config_hidden_units = hidden_units
        config_activation = activation
        config_loss = loss
        config_learning_rate = learning_rate
        config_batch_size = batch_size

        if hyperparam == "hidden_layers":
            config_hidden_layers = int(value)
        elif hyperparam == "hidden_units":
            config_hidden_units = int(value)
        elif hyperparam == "activation":
            config_activation = value
        elif hyperparam == "loss":
            config_loss = value
        elif hyperparam == "learning_rate":
            config_learning_rate = float(value)
        elif hyperparam == "batch_size":
            config_batch_size = int(value)

        model = _build_mlp_model(
            input_dim=train_features.shape[1],
            num_classes=num_classes,
            hidden_layers=config_hidden_layers,
            hidden_units=config_hidden_units,
            activation=config_activation,
        )

        loss_fn = MSE if config_loss == "mse" else CrossEntropy
        model.fit(
            train_features,
            train_targets,
            loss=loss_fn,
            epochs=epochs,
            batch_size=config_batch_size,
            learning_rate=config_learning_rate,
        )

        predictions = model.predict(test_features if task == "classification" else test_features)
        if task == "classification":
            train_pred = np.argmax(model.predict(train_features), axis=1)
            test_pred = np.argmax(predictions, axis=1)
            train_acc.append(accuracy_fn(train_pred, train_labels))
            test_acc.append(accuracy_fn(test_pred, test_labels))
            train_f1.append(macrof1_fn(train_pred, train_labels))
            test_f1.append(macrof1_fn(test_pred, test_labels))
        else:
            train_pred = np.ravel(model.predict(train_features))
            test_pred = np.ravel(predictions)
            train_mse.append(mse_fn(train_pred, train_labels))
            test_mse.append(mse_fn(test_pred, test_labels))

    result = {"values": list(values)}
    if task == "classification":
        result.update({
            "train_acc": train_acc,
            "test_acc": test_acc,
            "train_f1": train_f1,
            "test_f1": test_f1,
        })
    else:
        result.update({"train_mse": train_mse, "test_mse": test_mse})
    return result


def plot_mlp_hyperparams(
    train_features,
    train_labels,
    test_features,
    test_labels,
    hyperparam,
    values,
    task="classification",
    hidden_layers=1,
    hidden_units=64,
    activation="relu",
    loss="mse",
    epochs=500,
    batch_size=32,
    learning_rate=1e-3,
    filename_prefix=None,
):
    """Plot MLP metrics across hyperparameter values."""
    results = evaluate_mlp_hyperparams(
        train_features,
        train_labels,
        test_features,
        test_labels,
        hyperparam,
        values,
        task=task,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
        activation=activation,
        loss=loss,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )

    if task == "classification":
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required to plot hyperparameter curves. "
                "Install it with `pip install matplotlib`."
            ) from exc

        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

        # Accuracy subplot
        axes[0].plot(results["values"], results["train_acc"], marker="o", label="Train accuracy")
        axes[0].plot(results["values"], results["test_acc"], marker="o", label="Test accuracy")
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title(f"MLP metrics vs {hyperparam}")
        axes[0].grid(True)
        axes[0].legend()

        # Macro F1 subplot
        axes[1].plot(results["values"], results["train_f1"], marker="o", label="Train macro F1")
        axes[1].plot(results["values"], results["test_f1"], marker="o", label="Test macro F1")
        axes[1].set_ylabel("Macro F1")
        axes[1].set_xlabel(hyperparam)
        axes[1].grid(True)
        axes[1].legend()

        plt.tight_layout()
        if filename_prefix:
            filename = f"{filename_prefix}_{hyperparam}_multiplot.png"
            fig.savefig(filename, bbox_inches="tight")
        plt.show()
    else:
        _plot_curve(
            results["values"],
            [(results["train_mse"], "Train MSE"), (results["test_mse"], "Test MSE")],
            xlabel=hyperparam,
            ylabel="MSE",
            title=f"MLP MSE vs {hyperparam}",
            filename=f"{filename_prefix}_{hyperparam}_mse.png" if filename_prefix else None,
        )


def evaluate_kmeans_hyperparams(
    train_features,
    train_labels,
    test_features,
    test_labels,
    K_values,
    max_iters=100,
):
    """Evaluate KMeans classification metrics for different values of K."""
    # sanitize K_values: coerce to int, remove duplicates, sort
    try:
        K_list = list(dict.fromkeys([int(k) for k in K_values]))
    except Exception:
        K_list = [int(k) for k in K_values]
    K_list = [k for k in K_list if k >= 1]
    # don't allow K greater than number of samples
    max_K = train_features.shape[0]
    sanitized = []
    for k in K_list:
        if k > max_K:
            print(f"Warning: skipping K={k} because it's larger than number of samples ({max_K})")
            continue
        sanitized.append(k)
    K_list = sorted(sanitized)

    train_acc = []
    test_acc = []
    train_f1 = []
    test_f1 = []
    for K in K_list:
        model = KMeans(K=K, max_iters=max_iters)
        train_pred = model.fit(train_features, train_labels)
        test_pred = model.predict(test_features)

        train_acc.append(accuracy_fn(train_pred, train_labels))
        test_acc.append(accuracy_fn(test_pred, test_labels))
        train_f1.append(macrof1_fn(train_pred, train_labels))
        test_f1.append(macrof1_fn(test_pred, test_labels))

    return {
        "K_values": list(K_list),
        "train_acc": train_acc,
        "test_acc": test_acc,
        "train_f1": train_f1,
        "test_f1": test_f1,
    }


def plot_kmeans_hyperparams(
    train_features,
    train_labels,
    test_features,
    test_labels,
    K_values,
    max_iters=100,
    filename_prefix=None,
):
    """Plot KMeans accuracy and macro F1 versus the number of clusters K."""
    results = evaluate_kmeans_hyperparams(
        train_features,
        train_labels,
        test_features,
        test_labels,
        K_values,
        max_iters=max_iters,
    )

    _plot_curve(
        results["K_values"],
        [(results["train_acc"], "Train accuracy"), (results["test_acc"], "Test accuracy")],
        xlabel="K",
        ylabel="Accuracy (%)",
        title="KMeans accuracy vs number of clusters",
        filename=f"{filename_prefix}_kmeans_accuracy.png" if filename_prefix else None,
    )

    _plot_curve(
        results["K_values"],
        [(results["train_f1"], "Train macro F1"), (results["test_f1"], "Test macro F1")],
        xlabel="K",
        ylabel="Macro F1",
        title="KMeans macro F1 vs number of clusters",
        filename=f"{filename_prefix}_kmeans_f1.png" if filename_prefix else None,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        default="classification",
        type=str,
        help="classification / regression / clustering",
    )
    parser.add_argument(
        "--method",
        default="dummy_classifier",
        type=str,
        help="dummy_classifier / kmeans / mlp",
    )
    parser.add_argument(
        "--data_path",
        default="data/features.npz",
        type=str,
        help="path to your dataset CSV file",
    )
    parser.add_argument(
        "--K",
        type=int,
        default=27,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=7e-3,
        help="learning rate for methods with learning rate",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=100,
        help="max iters for methods which are iterative",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="train on whole training data and evaluate on the test data, "
             "otherwise use a validation set",
    )
    # Feel free to add more arguments here if you need!

    # Arguments for the MLP
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=32,
        help="number of units in each hidden layer for the MLP",
    )
    parser.add_argument(
        "--hidden_layers",
        type=int,
        default=1,
        help="number of hidden layers in the MLP",
    )
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "sigmoid", "tanh", "linear"],
        help="activation function for the MLP hidden layers",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="mse",
        choices=["mse", "crossentropy"],
        help="loss function to use for MLP training",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="number of training epochs for the MLP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for MLP training",
    )
    parser.add_argument(
        "--plot_mlp",
        action="store_true",
        help="Plot MLP metric curves instead of running a single training session",
    )
    parser.add_argument(
        "--plot_kmeans",
        action="store_true",
        help="Plot KMeans metric curves instead of running a single training session",
    )
    parser.add_argument(
        "--plot_hyperparam",
        type=str,
        default="hidden_units",
        choices=["hidden_layers", "hidden_units", "activation", "loss", "learning_rate", "batch_size"],
        help="MLP hyperparameter to sweep for plotting",
    )
    parser.add_argument(
        "--plot_values",
        type=str,
        default="16,32,64",
        help="Comma-separated hyperparameter values to sweep for MLP plotting",
    )
    parser.add_argument(
        "--plot_task",
        type=str,
        default=None,
        choices=["classification", "regression"],
        help="Task type to use for MLP plotting (if omitted, uses --task)",
    )
    parser.add_argument(
        "--plot_k_values",
        type=str,
        default="2,3,4,5",
        help="Comma-separated K values to sweep for KMeans plotting",
    )
    parser.add_argument(
        "--plot_max_iters",
        type=int,
        default=100,
        help="Max iterations for KMeans when plotting",
    )
    parser.add_argument(
        "--plot_filename_prefix",
        type=str,
        default=None,
        help="Optional filename prefix to save plot images",
    )

    args = parser.parse_args()
    main(args)
