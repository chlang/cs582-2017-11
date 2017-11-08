import numpy as np
import pcn

a = np.array([[1,1,1],[1,2,1],[2,1,1],[0,0,0],[1,0,0],[0,1,0]])

x = a[:, 0:2]
y = a[:, 2:]

p = pcn.pcn(x, y)
w = p.pcntrain(x, y, 0.25, 20)
p.confmat(x, y)


print(w)
# """"
# since we proved from the problem 1 that x+y=3/2 is separating vector
# from the several training of perceptron we get trained output value of w
# [[ 0.53248878]
#  [ 0.5161737 ]
#  [ 0.73048374]]
# w[0]x+w[1]y = w[3]
# => 0.5x+0.5y = 0.7
# => x + y = 7/5
# 7/5 ≈ 3/2
# => 14/10 ≈ 15/10
# So the perceptron returns almost same result which we found manually in #1 problem
# """"
