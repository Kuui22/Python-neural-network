import sys
import numpy as np
import matplotlib

# array: [1,2] Shape: (2,) Type: 1D Array, Vector
# array: [[1,5,6,2],[3,2,1,3]]  Shape: (2,4) Type: 2D Array, Matrix
# array: [[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]]]  Shape: (3, 2,4) Type: 3D Array
'''
matrix dot product =[1,2,3,4]  dot [9,8,7,6]   =  [1st row dot 1st column,1st row dot 2nd column, 1st row dot 3rd column, 1st row dot 4th column] 
                    [5,6,7,8]      [5,4,3,2]      [2nd row * 1st column,2nd *etc,etc,etc]
                    [9,1,2,3]      [1,2,3,4]      [3rd row * 1st column,...]
                    [4,5,6,7]      [5,6,7,8]      [4th row * 1st column,...]
'''

inputs:list[list[float]] = [[1, 2, 3, 2.5],[2.0, 5.0, -1.0, 2.0],[-1.5, 2.7, 3.3, -0.8]]
bias:list[float] = [2,3,0.5]
weights:list[list[float]] = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
neurons:int = len(weights)

bias2:list[float] = [-1,2,-0.5]
weights2:list[list[float]] = [[0.1, -0.14, 0.5],[-0.5, 0.12, -0.33],[-0.44, 0.73, -0.13]]

#dot of inputs with weights transposed. .T so that you can match the length of inputs with each column of weights
layer1_output = np.dot(inputs,np.array(weights).T) + bias
layer2_output = np.dot(layer1_output,np.array(weights2).T) + bias2
print(layer2_output)

exit(0)