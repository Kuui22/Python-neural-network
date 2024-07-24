import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence

#np.random.seed(0) legacy

seed_sequence = np.random.SeedSequence()
rs = RandomState(MT19937(seed_sequence))

X = [[1, 2, 3, 2.5],[2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]


class Layer_Dense:
    def __init__(self,n_inputs, n_neurons) -> None:
        self.weights = 0.10 * rs.randn(n_inputs, n_neurons) # makes a matrix where rows = n inputs and cols = n neurons
        self.biases = np.zeros((1, n_neurons)) # make an array long as the number of neurons (number of outputs)
    def forward(self, inputs) -> None:
        self.output = np.dot(inputs, self.weights) + self.biases #base dot operation + biases
    
    
layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2) # layer inputs must be the same as the layer before's outputs
layer3 = Layer_Dense(2,5)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
layer3.forward(layer2.output)
print(layer3.output)

exit(0)