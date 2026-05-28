import numpy as np

class Sigmoid:
    @staticmethod
    def forward(z):
        """
        applies sigmoid function element-wise
        arguments:
            z (np.array): pre-activation values, any shape
        return:
            (np.array): sigmoid(z), same shape as z
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    
    def gradient(z):
        """
        computes gradient of sigmoid
        arguments:
            z (np.array): pre-activation values, any shape
        return:
            (np.array): gradient of sigmoid at z, same shape as z
        """
        s = Sigmoid.forward(z)
        return s * (1.0 - s)

class ReLU:
    @staticmethod
    def forward(z):
        """
        applies ReLU element-wise.
        arguments:
            z (np.array): pre-activation values, any shape
        return:
            (np.array): relu(z), same shape as z
        """
        return np.maximum(0.0, z)

    @staticmethod
    def gradient(z):
        """
        computes gradient of ReLU
        arguments:
            z (np.array): pre-activation values, any shape
        return:
            (np.array): gradient of ReLU at z, same shape as z
        """
        return (z > 0).astype(float)
    
# Bonus: Implementation of tanh class function 
class Tanh:

    @staticmethod
    def forward(z):
        return np.tanh(z)

    @staticmethod
    def gradient(z):
        return 1.0 - np.tanh(z) ** 2


class Linear:

    @staticmethod
    def forward(z):
        return z

    @staticmethod
    def gradient(z):
        return np.ones_like(z)

