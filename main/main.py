import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
import matplotlib.pyplot as plt
# np.random.seed(0) legacy

seed_sequence = np.random.SeedSequence()
rs = RandomState(MT19937(seed_sequence))

#TODO:

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.weights = 0.10 * rs.randn(
            n_inputs, n_neurons
        )  # makes a matrix where rows = n inputs and cols = n neurons
        self.biases = np.zeros(
            (1, n_neurons)
        )  # make an array long as the number of neurons (number of outputs)

    def forward(self, inputs, activation=None) -> None:
        self.output = (
            np.dot(inputs, self.weights) + self.biases
        )  # base dot operation + biases
        # print(f"activation is:{activation}")
        if activation:
            self.output = activation.forward(self.output)


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)  # 0 if < 0 or x if > 0
        return self.output


X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]


layer1 = Layer_Dense(4, 5)
layer1.forward(X, activation=Activation_ReLU())
print(layer1.output)
