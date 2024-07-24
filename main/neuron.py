#neuron no imports
inputs:list[float] = [1, 2, 3, 2.5]
'''
weights1:list[float] = [0.2, 0.8, -0.5, 1.0]
weights2:list[float] = [0.5, -0.91, 0.26, -0.5]
weights3:list[float] = [-0.26, -0.27, 0.17, 0.87]
bias1:float = 2
bias2:float = 3
bias3:float = 0.5
'''
'''
output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
          inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
          inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]
print(output)
'''
bias:list[float] = [2,3,0.5]
weights = [[0.2, 0.8, -0.5, 1.0],[0.5, -0.91, 0.26, -0.5],[-0.26, -0.27, 0.17, 0.87]]
neurons = len(weights)
foroutput:list[float] = [0.0]* neurons 

for x in range(neurons):
   for i in range(len(inputs)):
       foroutput[x] += inputs[i]*weights[x][i]
   foroutput[x]+=bias[x]
print(foroutput)


exit(0)