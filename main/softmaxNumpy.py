import math
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],[8.9,-1.81,0.2],[1.41,1.051,0.026]]
e = math.e


exp_values = np.exp(layer_outputs) #exponential e for every value

norm_base = np.sum(layer_outputs, axis=1, keepdims=True) # sum all exponential values, axis = 0 for all columns, 1 for all rows. Keepdims to keep outputs dimensions

norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)  
print(norm_values)