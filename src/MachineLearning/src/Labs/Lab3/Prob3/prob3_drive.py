# Problem drive script 
#Problem 3 solution Group 8 

"""
This scrpit imports main module: Prob3 to train MLP  with two hidden layers.
As instructed in the questio we used Pima Indian  dataset

"""
import pylab as pl
import numpy as np
import Prob3 as mlp  # mlp is the main function in there. That is why we renamed the import as mlp 


pima = np.loadtxt('pima-indians-diabetes.data', delimiter=',')

# Randomize rows order
order = np.arange(pima.shape[0])
np.random.shuffle(order)
pima = pima[order,:]

train, traintargets = pima[ ::2, 7:8], pima[ ::2, 8: ]
valid, validtargets = pima[1::4, 7:8], pima[1::4, 8: ]
test,  testtargets  = pima[3::4, 7:8], pima[3::4, 8: ]

# Train  

net = mlp.mlp(train,traintargets,nhidden=1,outtype='linear')
net.earlystopping(train,traintargets,valid,validtargets,0.2)
net.confmat(test, testtargets)



