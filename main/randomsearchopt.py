"""
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
"""
# random search
"""
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
"""