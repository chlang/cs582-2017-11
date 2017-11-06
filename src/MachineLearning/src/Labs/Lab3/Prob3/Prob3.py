import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
    def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        """ Constructor """
        # Set up network size
        self.nin = inputs.shape[1]
        self.nout = targets.shape[1]
        self.ndata = inputs.shape[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype

        # Initialise network
        '''
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        '''
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        self.weights3 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        
        '''
        print("init in: ", inputs.shape)
        print("init w1: ", self.weights1.shape)        
        print("init w2: ", self.weights2.shape)
        print("init w3: ", self.weights3.shape)
        '''
        
    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
    
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print( count )
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print( "Stopped", new_val_error,old_val_error1, old_val_error2 )
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        #change = np.arange(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
        ''' '''
        updatew3 = np.zeros((np.shape(self.weights3))) 
        
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print( "Iteration: ",n, " Error: ",error )    

            # Different types of output neurons
            if self.outtype == 'linear':
                deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
                deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print( "error" )
            
            '''
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
            '''
            '''
            print("deltao: ", deltao.shape)
            print("h2: ", self.hidden2.shape)
            '''
            deltah2 = self.hidden2*self.beta*(1.0-self.hidden2)*(np.dot(deltao,self.weights3)) 
            '''
            print("deltah2: ", deltah2.shape)
            print("h1: ", self.hidden1.shape)
            '''
            deltah1 = self.hidden1*self.beta*(1.0-self.hidden1)*(np.dot(deltah2,self.weights2.transpose()))
            
            '''
            print("h1: ", np.shape(self.hidden1*self.beta*(1.0-self.hidden1)) ) 
            print("w2: ", self.weights2.shape)  
            print("deltah1: ", deltah1.shape)  
            print("w2*deltah2: ", np.shape(np.dot(deltah2,self.weights2.transpose()))  )
            '''
            '''
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            '''
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah1[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden1),deltah2)) + self.momentum*updatew2
            '''
            print("h2: ", self.hidden2.transpose().shape)
            print("do: ", deltao.shape)   
            print("uw3: ", updatew3.shape) 
            '''
            updatew3 = eta*(np.dot(self.hidden2.transpose(),deltao).transpose()) + self.momentum*updatew3
            
            '''     
            print("inputs: ", inputs.shape)
            print("w1: ", self.weights1.shape)
            print("uw1: ", updatew1.shape)  
            print("w2: ", self.weights2.shape)
            print("uw2: ", updatew2.shape)  
            print("w3: ", self.weights3.shape)
            print("uw3: ", updatew3.shape) 
            '''
            self.weights1 -= updatew1
            self.weights2 -= updatew2
            ''' '''
            self.weights3 -= updatew3
            
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        '''
        self.hidden = np.dot(inputs,self.weights1);                            
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((inputs.shape[0],1))),axis=1)
        '''
        self.hidden1 = np.dot(inputs,self.weights1);                            
        self.hidden1 = 1.0/(1.0+np.exp(-self.beta*self.hidden1))
        self.hidden1 = np.concatenate((self.hidden1,-np.ones((inputs.shape[0],1))),axis=1)
        
        self.hidden2 = np.dot(self.hidden1,self.weights2);                            
        self.hidden2 = 1.0/(1.0+np.exp(-self.beta*self.hidden2))
        
        
        '''
        outputs = np.dot(self.hidden,self.weights2);
        '''
        '''
        print("in:", inputs.shape)
        print("w1:", self.weights1.shape)
        print("h1:", self.hidden1.shape)
        print("w2:", self.weights2.shape)
        print("h2:", self.hidden2.shape)
        print("w3:", self.weights3.shape)
        '''
        
        outputs = np.dot(self.hidden2,self.weights3.transpose());
       
        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print( "error" )

    def confmat(self,inputs,targets):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)

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

        print( "Confusion matrix is:" )
        print( cm )
        print( "Percentage Correct: ",np.trace(cm)/np.sum(cm)*100 )
