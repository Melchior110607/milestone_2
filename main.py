import argparse
import numpy as np

from src.methods.dummy_methods import DummyClassifier
from src.methods.mlp import MLP
from src.losses import MSE
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

    train_features = append_bias_term(train_features)
    test_features = append_bias_term(test_features)

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
            train_predictions = method_obj.fit(
                train_features,
                train_targets,
                loss=MSE,
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
        default=1,
        help="number of clusters datapoints used for kmeans",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-5,
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
        default=64,
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
        "--epochs",
        type=int,
        default=5000,
        help="number of training epochs for the MLP",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=20,
        help="batch size for MLP training",
    )

    args = parser.parse_args()
    main(args)
