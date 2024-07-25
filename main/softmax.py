import math

layer_outputs = [4.8, 1.21, 2.385]
e = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(e**output)
    
    
print(exp_values)

norm_base = sum(exp_values)

norm_values =[]

for value in exp_values:  # create normalization of values (value 1 is 89% of sum (0.89), value 2 is 2.4%(0.024), etc...)
    norm_values.append(value / norm_base)
    
print(norm_values)
print(sum(norm_values))