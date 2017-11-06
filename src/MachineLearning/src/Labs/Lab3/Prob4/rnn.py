import numpy as np

class rnn:
    """ A Recurrent Neural Network Perceptron"""

    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.ndata = np.shape(inputs)[0]
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        sn = np.sqrt(self.nin + 1 + self.nout)
        self.weights1i = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/sn
        self.weights1o = (np.random.rand(self.nout, self.nhidden)-0.5)*2/sn
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):

        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)

        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000

        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print(count)
            self.train(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.fwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)

        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error

    def train(self,inputs,targets,eta,niterations):
        """ Train the thing """
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        #change = range(self.ndata)

        updatew1i = np.zeros((np.shape(self.weights1i)))
        updatew1o = np.zeros((np.shape(self.weights1o)))
        updatew2 = np.zeros((np.shape(self.weights2)))

        for n in range(niterations):

            self.outputs = self.fwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error)

                # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata
            else:
                print("error")

            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))

            os = np.roll(self.outputs, -1, axis=0)
            os[0] = np.zeros((1, self.nout))

            updatew1i = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1i
            updatew1o = eta*(np.dot(np.transpose(os), deltah[:,:-1])) + self.momentum*updatew1o
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1i -= updatew1i
            self.weights1o -= updatew1o
            self.weights2 -= updatew2


    def fwd(self,inputs):
        """ Run the network forward """

        ndata = np.shape(inputs)[0]
        self.hidden = np.zeros((ndata, self.nhidden+1))
        outputs = np.zeros((ndata, self.nout))

        for i in range(ndata):
            hs = np.dot(inputs[i], self.weights1i)
            if i > 0:
                hs += np.dot(outputs[i-1], self.weights1o)
            hs = 1.0/(1.0+np.exp(-self.beta * hs))
            hs = np.reshape(hs, (1, self.nhidden))
            self.hidden[i] = np.concatenate((hs, -np.ones((1, 1))), axis=1)
            outputs[i] = np.dot(self.hidden[i], self.weights2)

        # Different types of output neurons
        if self.outtype == 'linear':
            return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.fwd(inputs)

        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))

        print("Confusion matrix is:")
        print(cm)
        print("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)