import numpy as np
import pcn

a = np.array([[1,1,1],[1,2,1],[2,1,1],[0,0,0],[1,0,0],[0,1,0]])

x = a[:, 0:2]
y = a[:, 2:]

p = pcn.pcn(x, y)
w = p.pcntrain(x, y, 0.25, 20)
p.confmat(x, y)


print(w)
