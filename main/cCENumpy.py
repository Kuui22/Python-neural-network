import numpy as np
'''
so in this example, we have 3 types of class : 0 , 1 , 2
each element in a row is the probability of that class
so for class 0 in the first row of outputs the probability is 0.7
in the second 0.1
in the third 0.02
for class 1 in the first 0.1, the second 0.5, etc...
class targets is like this:
in the first select the probability of my class (0) which equals to 0.7
in the second select the probability of my class (1) which equals to 0.5
etc...
'''
softmax_outputs = np.array([[0.70, 0.1, 0.2],
                            [0.1, 0.5, 0.4,],
                            [0.02, 0.9, 0.08]])


class_targets=[0,1,1] #this just means: in the first output the correct class was 0, in the second 1, and in the third 1

#this is just simplification because of how classification works
# range outputs a list of numbers equal to the int provided. so in this case range=[0,1,2]
negative_log= (-np.log(softmax_outputs[range(len(softmax_outputs)),class_targets])) 
average_loss = np.mean(negative_log)
print(negative_log)
#needs clip