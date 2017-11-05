# Code from Chapter 3 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np


class pcn:
    """ A basic Perceptron (the same pcn.py except with the weights printed
    and it does not reorder the inputs)"""

    def __init__(self, inputs, targets):
        """ Constructor """
        # Set up network size
        if np.ndim(inputs) > 1:
            self.nIn = np.shape(inputs)[1]
        else:
            self.nIn = 1

        if np.ndim(targets) > 1:
            self.nOut = np.shape(targets)[1]
        else:
            self.nOut = 1

        self.nData = np.shape(inputs)[0]

        # Initialise network
        self.weights = np.random.rand(self.nIn + 1, self.nOut) * 0.1 - 0.05

    # 3/3/17 For sequential learning - just assume we are taking one row of the input vector at a time.  This way
    # the code change will be miminimal. Let's call it inputs_inter.

    def pcntrain(self, inputs, targets, eta, nIterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)

        # Training
        change = list(range(self.nData))

        for n in range(nIterations):

            for m in range(self.nData):
                inputs_seq = np.array([inputs[m]])
                targets_seq = np.array([targets[m]])
                self.activations = self.pcnfwd(self.weights, inputs_seq)
                self.weights -= eta * np.dot(np.transpose(inputs_seq), self.activations - targets_seq)
                print("Iteration: ", n)
                print("Interactive steps m: ", m)
                print(self.weights)
                activations = self.pcnfwd(inputs_seq)
                print("Final outputs after each input vector are:")
                print(activations)
                # return self.weights

    def pcnfwd(self, inputs):
        """ Run the network forward """

        # Compute activations
        activations = np.dot(inputs, self.weights)

        # Threshold the activations
        return np.where(activations > 0, 1, 0)

    def confmat(self, inputs, targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs, -np.ones((self.nData, 1))), axis=1)
        outputs = np.dot(inputs, self.weights)

        nClasses = np.shape(targets)[1]

        if nClasses == 1:
            nClasses = 2
            outputs = np.where(outputs > 0, 1, 0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs, 1)
            targets = np.argmax(targets, 1)

        cm = np.zeros((nClasses, nClasses))
        for i in range(nClasses):
            for j in range(nClasses):
                cm[i, j] = np.sum(np.where(outputs == i, 1, 0) * np.where(targets == j, 1, 0))

        print(cm)
        print(np.trace(cm) / np.sum(cm))
