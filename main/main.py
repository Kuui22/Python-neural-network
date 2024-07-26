import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
# import matplotlib
# np.random.seed(0) legacy

seed_sequence = np.random.SeedSequence()
rs = RandomState(MT19937(seed_sequence))


# TODO:backpropagation


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype="uint8")
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


class Layer_Dense:
    def __init__(self, n_inputs, n_neurons) -> None:
        #He Initialization
        self.weights = np.sqrt(2.0 / n_inputs) * rs.randn(n_inputs, n_neurons)  # makes a matrix where rows = n inputs and cols = n neurons
        self.biases = np.zeros((1, n_neurons))  # make an array long as the number of neurons (number of outputs)
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs

    def forward(self, inputs, activation=None) -> None:
        #save inputs for backwards
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases  # base dot operation + biases
        # print(f"activation is:{activation}")
        if activation:
            self.output = activation.forward(self.output)
    def backward(self, dvalues):
        
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        
        self.clip_gradients()

    def clip_gradients(self, max_norm=5.0):
        #This norm is a measure of the magnitude of the gradient vector.
        norm = np.linalg.norm(self.dweights)
        if norm > max_norm:
            self.dweights = self.dweights * max_norm / norm
        norm = np.linalg.norm(self.dbiases)
        if norm > max_norm:
            self.dbiases = self.dbiases * max_norm / norm


class Activation_ReLU:
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        self.output = np.maximum(0, inputs)  # 0 if < 0 or x if > 0
        return self.output

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] = 0


class Activation_Softmax:
    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # exponential of e for every (value in row - max value of row[to avoid overflow])
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # value in row / sum of all values in row
        self.output = probabilities
        return self.output

    def backward(self, dvalues):
        # Create uninitialized array
        self.dinputs = np.empty_like(dvalues)

        # Enumerate outputs and gradients
        for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
            # Flatten output array
            single_output = single_output.reshape(-1, 1)
            # Calculate Jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)

            # Calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


"""
example shape=1
scalar class values
[0,1,1]
example  shape=2
one hot encoded
[[1,0,0],
 [0,0,1],
 [1,0,0]]
so everything that isn't the target gets nulled
"""


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]  # grab the element of row x of range samples equal to row x of y_true
        elif len(y_true.shape) == 2:  # if hot encoded, its a matrix
            correct_confidences == np.sum(y_pred_clipped * y_true, axis=1)  # multiply each element in a row of y_pred * element in the same position of y_true

        negative_log_likelihoods = -np.log(correct_confidences)  # simplified cross entropy
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)
        # Number of labels in every sample
        # We'll use the first sample to count them
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]
        
        dvalues_clipped = np.clip(dvalues, 1e-7, 1 - 1e-7)

        # Calculate gradient
        self.dinputs = -y_true / dvalues_clipped  # d/dx (ln x) = 1/x derivative of natural log is 1/x
        # Normalize gradient
        self.dinputs = self.dinputs / samples


def predict(softmax_outputs, targets):
    predictions = np.argmax(softmax_outputs, axis=1)
    accuracy = np.mean(predictions == targets)  # where predictions[x] == targets [x]
    return accuracy


X, y = spiral_data(points=100, classes=3)


dense1 = Layer_Dense(2, 256)# n_inputs = number of dimensions of input
dense2 = Layer_Dense(256, 256)
dense3 = Layer_Dense(256, 128)
dense4 = Layer_Dense(128, 3)  # output layer neurons = number of classes
'''
dense1.forward(X, activation=Activation_ReLU())
dense2.forward(dense1.output, activation=Activation_Softmax())

# print(dense2.output[:5])

loss_function = Loss_CategoricalCrossentropy()
loss = loss_function.calculate(dense2.output, y)
accuracy = predict(dense2.output, y)
print(f"Loss:{loss}")
print(f"Accuracy:{accuracy}")

lowest_loss = loss
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()
'''
# random search
'''
for iteration in range(100000):
    dense1.weights += 0.05 * rs.randn(dense1.n_inputs, dense1.n_neurons)
    dense1.biases += 0.05 * rs.randn(1, dense1.n_neurons)
    dense2.weights += 0.05 * rs.randn(dense2.n_inputs, dense2.n_neurons)
    dense2.biases += 0.05 * rs.randn(1, dense2.n_neurons)

    dense1.forward(X, activation=Activation_ReLU())
    dense2.forward(dense1.output, activation=Activation_Softmax())
    loss = loss_function.calculate(dense2.output, y)
    accuracy = predict(dense2.output, y)

    # print(f"Loss:{loss},Accuracy:{accuracy}")

    if loss < lowest_loss:
        print(f"New record:Iteration:{iteration}, Loss:{loss}, Accuracy:{accuracy}")
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()
'''
loss_function = Loss_CategoricalCrossentropy()
lowest_loss = 999999
learning_rate = 1e-5
activation1=Activation_ReLU()
activation2=Activation_ReLU()
activation3=Activation_ReLU()
activation4=Activation_Softmax()
epochs = 1000000
for epoch in range(epochs):
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    dense3.forward(activation2.output)
    activation3.forward(dense3.output)
    dense4.forward(activation3.output)
    activation4.forward(dense4.output)
    loss = loss_function.calculate(activation4.output, y)
    accuracy = predict(activation4.output, y)

    # print(f"Loss:{loss},Accuracy:{accuracy}")
    if loss < lowest_loss:
        #print(f"New record:Iteration:{epoch}, Loss:{loss}, Accuracy:{accuracy}")
        lowest_loss = loss
    if epoch % 1000 == 0:
        print(f'Epoch {epoch}, Loss: {loss}, Accuracy: {accuracy}, Lowest Loss:{lowest_loss}')
        
    loss_function.backward(activation4.output,y)
    dense4.backward(loss_function.dinputs)
    activation3.backward(dense4.dinputs)
    dense3.backward(activation3.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)
    
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases
    dense3.weights -= learning_rate * dense3.dweights
    dense3.biases -= learning_rate * dense3.dbiases
    dense4.weights -= learning_rate * dense4.dweights
    dense4.biases -= learning_rate * dense4.dbiases
