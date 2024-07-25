'''
# natural log x is log of x in base e (euler)
a log in base e means:
e ** x = y : e elevated to what x equals y

so a log means
which number the base has to be elevated to, to get the target number [logx(y)] where x = base y = target

'''
import numpy as np
import math

y = 5

x = np.log(y)
print(x)

expected_y = math.e ** x

print(expected_y)
