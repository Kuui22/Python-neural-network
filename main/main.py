import numpy as np
from numpy.random import MT19937, RandomState, SeedSequence
# import matplotlib
# np.random.seed(0) legacy

seed_sequence = np.random.SeedSequence()
rs = RandomState(MT19937(seed_sequence))

#TODO: add batch normalization and dropout. also what if you switch dataset in the middle of training?

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

def generate_new_dataset(points,classes):
    X, y = spiral_data(points=points,classes=classes)
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8  # avoid dividing by zero with addiction
    X_normalized = (X - X_mean) / X_std
    return X_normalized,y
    

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
        if self.activation:
            dvalues = self.activation.backward(dvalues)
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradient on values
        self.dinputs = np.dot(dvalues, self.weights.T)
        # self.clip_gradients()

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
        print(f"layer ninputs:{layer.n_inputs},neurons:{layer.n_neurons}")
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

# L2 regularization (weight decay)
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

spiral_points = 100
spiral_classes = 3
X_normalized, y = generate_new_dataset(spiral_points,spiral_classes)
X_val_normalized, y_val =generate_new_dataset(spiral_points,spiral_classes)


lowest_loss = 999999
lowest_val_loss = 999999
initial_learning_rate = 1e-4
# decay rate of the learning rate
decay_rate = 0.01
decay_steps = 500

loss_function = Loss_CategoricalCrossentropy()
# Add L2 regularization, this gets applied to weights to avoid vanishing
l2_lambda = 1e-4

network = create_network(n_input=2,n_output=3,hidden_activation=Activation_Leaky_ReLU,n_layers=4)
optimizer = AdamOptimizer(network, learning_rate=initial_learning_rate)
epochs = 4000

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

        #forward pass
        forward_output = network_forward(network,X_batch)
        loss = loss_function.calculate(forward_output, y_batch)
        accuracy = predict(forward_output, y_batch)

        epoch_loss += loss #add loss/acc of batch to total loss
        epoch_accuracy += accuracy
        num_batches += 1

        #backward pass
        loss_function.backward(forward_output, y_batch)
        backward_output = network_backward(network,loss_function.dinputs)
        
        #opts
        network_L2_regularization(network,l2_lambda=l2_lambda)
        optimizer.update()
        network_clip_gradients(network)

    epoch_loss /= num_batches #divide loss/acc per number of batches used
    epoch_accuracy /= num_batches

    # Learning rate decay
    learning_decay(initial_learning_rate=initial_learning_rate,optimizer=optimizer,decay_rate=decay_rate,epoch=epoch,decay_steps=decay_steps)

    if epoch_loss < lowest_loss:
        lowest_loss = epoch_loss

    if epoch % 100 == 0:
        val_output = network_forward(network,X_val_normalized)
        val_loss = loss_function.calculate(val_output,y_val)
        val_accuracy = predict(val_output, y_val)
        if(val_loss < lowest_val_loss):
            lowest_val_loss = val_loss
        print(f"Epoch {epoch}, Loss: {epoch_loss}, Accuracy: {epoch_accuracy}, Lowest Loss:{lowest_loss}, Val Loss:{val_loss}, Val acc:{val_accuracy} LVL{lowest_val_loss}")
        
