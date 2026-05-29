import numpy as np

class MSE:
    @staticmethod
    def loss(y_true, y_pred):
        """
        computes MSE loss
        arguments:
            y_true (np.array): ground truth values, shape (N, C) or (N,)
            y_pred (np.array): predicted values, shape (N, C) or (N,)
        return:
            (float): scalar MSE loss
        """
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def gradient(y_true, y_pred):
        """
        computes gradient of MSE loss 
        arguments:
            y_true (np.array): ground truth, shape (N, C) or (N,)
            y_pred (np.array): predictions, shape (N, C) or (N,)
        return:
            (np.array): gradient, same shape as y_pred
        """
        N = y_true.shape[0]
        return (2.0 / N) * (y_pred - y_true)
    
#Bonus: Implementation of CrossEntropy

class CrossEntropy:
    @staticmethod
    def _softmax(z):
        """
        Apply the softmax function to logits.
        """
        z = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    @staticmethod
    def loss(y_true, y_pred):
        """
        computes Cross-Entropy loss
        arguments:
            y_true (np.array): one-hot encoded truth, shape (N, C)
            y_pred (np.array): predicted logits, shape (N, C)
        return:
            (float): scalar cross-entropy loss
        """
        y_prob = CrossEntropy._softmax(y_pred)
        y_prob = np.clip(y_prob, 1e-12, 1.0 - 1e-12)
        N = y_true.shape[0]
        return -np.sum(y_true * np.log(y_prob)) / N

    @staticmethod
    def gradient(y_true, y_pred):
        """
        computes gradient of Cross-Entropy loss with softmax
        arguments:
            y_true (np.array): one-hot encoded truth, shape (N, C)
            y_pred (np.array): predicted logits, shape (N, C)
        return:
            (np.array): gradient, same shape as y_pred
        """
        y_prob = CrossEntropy._softmax(y_pred)
        N = y_true.shape[0]
        return (y_prob - y_true) / N

