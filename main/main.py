import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
# import matplotlib
# np.random.seed(0) legacy

seed_sequence = np.random.SeedSequence()
rs = RandomState(MT19937(seed_sequence))


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
    def __init__(self, n_inputs, n_neurons, activation=None) -> None:
        # He Initialization
        self.weights = np.sqrt(2.0 / n_inputs) * rs.randn(n_inputs, n_neurons)  # makes a matrix where rows = n inputs and cols = n neurons
        self.biases = np.zeros((1, n_neurons))  # make an array long as the number of neurons (number of outputs)
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.activation = activation

        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)

    def forward(self, inputs) -> None:
        # save inputs for backwards
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases  # base dot operation + biases
        # print(f"activation is:{activation}")
        if self.activation:
            self.output = self.activation.forward(self.output)

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        # self.clip_gradients()
        if self.activation:
            self.dinputs = self.activation.backward(self.dinputs)

    def clip_gradients(self, max_norm=9.0):
        # This norm is a measure of the magnitude of the gradient vector.
        norm = np.linalg.norm(self.dweights)
        if norm > max_norm:
            self.dweights = self.dweights * max_norm / norm
        norm = np.linalg.norm(self.dbiases)
        if norm > max_norm:
            self.dbiases = self.dbiases * max_norm / norm

    def get_parameters(self):
        return [self.weights, self.biases]

    def get_gradients(self):
        return [self.dweights, self.dbiases]


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
        
        return self.dinputs


class Activation_Leaky_ReLU:
    def __init__(self, alpha=0.01) -> None:
        self.alpha = alpha

    def forward(self, inputs):
        # Remember input values
        self.inputs = inputs
        # np.where(condition, if_true, if_false)
        self.output = np.where(inputs > 0, inputs, self.alpha * inputs)  # input if < 0 or input*alpha if < 0
        return self.output

    def backward(self, dvalues):
        # Since we need to modify original variable,
        # let's make a copy of values first
        self.dinputs = dvalues.copy()

        # Zero gradient where input values were negative
        self.dinputs[self.inputs <= 0] *= self.alpha
        
        return self.dinputs


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
            
        return self.dinputs


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


class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y_true]  # grab the element of row x of range samples equal to row x of y_true
        elif len(y_true.shape) == 2:  # if hot encoded, its a matrix
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)  # multiply each element in a row of y_pred * element in the same position of y_true

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

#create a network collection
def create_network(n_input, n_output, hidden_activation, n_layers, neurons_per_layer=64):
    layers = []

    for i in range(n_layers):
        if i == 0:
            # first layer with n input
            layer = Layer_Dense(n_inputs=n_input, n_neurons=neurons_per_layer, activation=hidden_activation())
        elif i == n_layers - 1:
            # last layer
            layer = Layer_Dense(n_inputs=layers[-1].n_neurons, n_neurons=n_output, activation=Activation_Softmax())
        else:
            # hidden layers
            layer = Layer_Dense(n_inputs=layers[-1].n_neurons, n_neurons=neurons_per_layer, activation=hidden_activation())

        layers.append(layer)

    return layers

#take input and then output of previous layer
def network_forward(network, inputs):
    prev_input = inputs
    for layer in network:
        layer.forward(prev_input)
        prev_input = layer.output

    return network[-1].output

#take loss dinputs and then dinputs of next layer
def network_backward(network, loss_dinputs):
    prev_dinputs = loss_dinputs
    for layer in reversed(network):
        layer.backward(prev_dinputs)
        prev_dinputs = layer.dinputs

def network_clip_gradients(network):
    for layer in network:
        layer.clip_gradients()

def network_L2_regularization(network,l2_lambda):
    for layer in network:
            layer.dweights += 2 * l2_lambda * layer.weights
            
def learning_decay(initial_learning_rate,optimizer,decay_rate,epoch,decay_steps):
    current_learning_rate = initial_learning_rate * (1.0 / (1.0 + decay_rate * epoch / decay_steps))
    optimizer.learning_rate = current_learning_rate

class AdamOptimizer:
    def __init__(self, layers, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.layers = layers
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m_weights = [np.zeros_like(layer.weights) for layer in self.layers]
        self.v_weights = [np.zeros_like(layer.weights) for layer in self.layers]
        self.m_biases = [np.zeros_like(layer.biases) for layer in self.layers]
        self.v_biases = [np.zeros_like(layer.biases) for layer in self.layers]
        self.t = 0

    def update(self):
        self.t += 1
        lr_t = self.learning_rate * np.sqrt(1 - self.beta2**self.t) / (1 - self.beta1**self.t)

        for i, layer in enumerate(self.layers):
            # Update weights
            self.m_weights[i] = self.beta1 * self.m_weights[i] + (1 - self.beta1) * layer.dweights
            self.v_weights[i] = self.beta2 * self.v_weights[i] + (1 - self.beta2) * (layer.dweights**2)
            m_hat = self.m_weights[i] / (1 - self.beta1**self.t)
            v_hat = self.v_weights[i] / (1 - self.beta2**self.t)
            layer.weights -= lr_t * m_hat / (np.sqrt(v_hat) + self.epsilon)

            # Update biases
            self.m_biases[i] = self.beta1 * self.m_biases[i] + (1 - self.beta1) * layer.dbiases
            self.v_biases[i] = self.beta2 * self.v_biases[i] + (1 - self.beta2) * (layer.dbiases**2)
            m_hat_bias = self.m_biases[i] / (1 - self.beta1**self.t)
            v_hat_bias = self.v_biases[i] / (1 - self.beta2**self.t)
            layer.biases -= lr_t * m_hat_bias / (np.sqrt(v_hat_bias) + self.epsilon)


X, y = spiral_data(points=100, classes=3)
# Normalize input data
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-8  # avoid dividing by zero
X_normalized = (X - X_mean) / X_std

lowest_loss = 999999
initial_learning_rate = 2e-7
# decay rate of the learning rate
decay_rate = 0.01
decay_steps = 500

loss_function = Loss_CategoricalCrossentropy()


dense1 = Layer_Dense(2, 128)  # n_inputs = number of dimensions of input
dense2 = Layer_Dense(128, 64)
dense3 = Layer_Dense(64, 32)
dense4 = Layer_Dense(32, 3)  # output layer neurons = number of classes

# Add L2 regularization, this gets applied to weights to avoid vanishing
l2_lambda = 1e-4

optimizer = AdamOptimizer([dense1, dense2, dense3, dense4], learning_rate=initial_learning_rate)

activation1 = Activation_Leaky_ReLU()
activation2 = Activation_Leaky_ReLU()
activation3 = Activation_Leaky_ReLU()
activation4 = Activation_Softmax()
epochs = 100000

# Implement mini-batch processing
batch_size = 32

for epoch in range(epochs):
    # Shuffle the data
    indices = np.arange(X_normalized.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X_normalized[indices]
    y_shuffled = y[indices]

    epoch_loss = 0
    epoch_accuracy = 0
    num_batches = 0

    # batch training
    for i in range(0, X_normalized.shape[0], batch_size):
        X_batch = X_shuffled[i : i + batch_size]
        y_batch = y_shuffled[i : i + batch_size]

        # Forward pass
        dense1.forward(X_batch)
        activation1.forward(dense1.output)
        dense2.forward(activation1.output)
        activation2.forward(dense2.output)
        dense3.forward(activation2.output)
        activation3.forward(dense3.output)
        dense4.forward(activation3.output)
        activation4.forward(dense4.output)
        loss = loss_function.calculate(activation4.output, y_batch)
        accuracy = predict(activation4.output, y_batch)

        epoch_loss += loss
        epoch_accuracy += accuracy
        num_batches += 1

        loss_function.backward(activation4.output, y_batch)
        dense4.backward(loss_function.dinputs)
        activation3.backward(dense4.dinputs)
        dense3.backward(activation3.dinputs)
        activation2.backward(dense3.dinputs)
        dense2.backward(activation2.dinputs)
        activation1.backward(dense2.dinputs)
        dense1.backward(activation1.dinputs)
        # L2 regularization (weight decay)
        for layer in [dense1, dense2, dense3, dense4]:
            layer.dweights += 2 * l2_lambda * layer.weights

        optimizer.update()

        dense1.clip_gradients()
        dense2.clip_gradients()
        dense3.clip_gradients()
        dense4.clip_gradients()

    epoch_loss /= num_batches
    epoch_accuracy /= num_batches

    # Learning rate decay
    current_learning_rate = initial_learning_rate * (1.0 / (1.0 + decay_rate * epoch / decay_steps))
    optimizer.learning_rate = current_learning_rate

    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Lowest Loss:{lowest_loss}")


"""
    dense1.weights -= learning_rate * dense1.dweights
    dense1.biases -= learning_rate * dense1.dbiases
    dense2.weights -= learning_rate * dense2.dweights
    dense2.biases -= learning_rate * dense2.dbiases
    dense3.weights -= learning_rate * dense3.dweights
    dense3.biases -= learning_rate * dense3.dbiases
    dense4.weights -= learning_rate * dense4.dweights
    dense4.biases -= learning_rate * dense4.dbiases
"""
