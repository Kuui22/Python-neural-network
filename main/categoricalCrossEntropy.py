import math

softmax_output = [0.7, 0.1, 0.2]
target_class = 0
target_output = [1,0,0] # one hot encoding

loss = -(target_output[0]*math.log(softmax_output[0]) + #cross entropy is the negative sum of the target value times the log of the predicted value for each of the values in the distribution
         target_output[1]*math.log(softmax_output[1]) +
         target_output[2]*math.log(softmax_output[2]))


print(loss)
loss = -math.log(softmax_output[0])
print(loss)