import sys
import numpy as np
import matplotlib

# array: [1,2] Shape: (2,) Type: 1D Array, Vector
# array: [[1,5,6,2],[3,2,1,3]]  Shape: (2,4) Type: 2D Array, Matrix
# array: [[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]],[[1,5,6,2],[3,2,1,3]]]  Shape: (3, 2,4) Type: 3D Array


inputs:list[float] = [1, 2, 3, 2.5]
bias:list[float] = [2,3,0.5]
weights:list[list[float]] = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
neurons:int = len(weights)

output = np.dot(weights,inputs) + bias
print(output)

exit(0)