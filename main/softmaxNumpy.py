import math
import numpy as np

layer_outputs = [[4.8, 1.21, 2.385],[8.9,-1.81,0.2],[1.41,1.051,0.026]]
e = math.e
exp_values = np.exp(layer_outputs) #exponential e for every value

norm_base = sum(exp_values) # sum all exponential values

norm_values = exp_values / np.sum(exp_values) # create normalization of values (value 1 is 89% of sum (0.89), value 2 is 2.4%(0.024), etc...)

print(norm_values)
print(sum(norm_values))