import numpy as np
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


class NeuralNetwork:
    def __init__(self, structure, learning_rate = 0.1, activation_func = 'ReLu', debug = False, bias_output = False,early_stop = False) -> None:         # Initialize weights and biases
        print('Initializing..')

        start_time = time.time()
        self.W = []
        self.B = []
        self.size = structure
        self.alpha = learning_rate
        self.act = activation_func
        self.debug = debug
        self.bias_on_output = bias_output
        self.loss_array = []
        self.stop = early_stop

        # Initialising weights
        for i in range(len(self.size)-1):
            w = np.random.randn(self.size[i],self.size[i+1])
            self.W.append(w)
        
        # Initialising biases
        for i in range(len(self.size)-2):
            b = np.random.uniform(-1,1,self.size[i+1])                
            self.B.append(b)
        if bias_output:
            b_out = np.random.uniform(-1,1,self.size[-1])
            self.B.append(b_out)

            
        end_time = time.time() - start_time
        print("Initializing done. {:2.0f}m{:2.0f}s".format(end_time//60, end_time%60))


    def act_func(self,X):                       # Add more functions
        if self.act == 'ReLu':
            return np.maximum(0,X)
        if self.act == 'Sigmoid':
            return 1.0 / (1 + np.exp(-X))


    def act_func_dif(self,X):
        if self.act == 'ReLu':
            return np.where(X > 0,1,0)

        if self.act == 'Sigmoid':
            return self.act_func(X) * (1 - self.act_func(X))
            # return X * (1 - X)



    def fit(self,X,Y,epochs = 100,update = 10, tol = 0.05, patience = 5):
        self.epochs = epochs
        self.update = update
        best_loss = None
        patience_count = 0

        # FP, loop iterations/epochs
        for epoch in np.arange(0,epochs):
            for (input,target) in zip(X,Y):         # Loop through training data, matching input with output
                self.train(input,target)
            if epoch == 0 or (epoch+1)%update == 0:
                loss = self.calculate_loss(X,Y)
                if self.stop:
                    if best_loss is None:
                        best_loss = loss
                    else:
                        if loss < best_loss:
                            best_loss = loss
                            patience_count = 0
                        elif abs(loss - best_loss) <= tol*best_loss:
                            patience_count += 1
                    
                        if patience_count >= patience:
                            print("[INFO] Early stopping at epoch={}, best_loss={:.7f}".format(epoch + 1, best_loss))
                            break
                        elif loss > best_loss*1.1:
                            print("[INFO] Early stopping (loss getting higher) at epoch={}, best_loss={:.7f}".format(epoch + 1, best_loss))
                            break
                else:
                    if best_loss is None or loss < best_loss:
                        best_loss = loss
                print("[INFO] epoch={}, loss={:.7f}".format(epoch + 1,loss))

        if self.debug:
            print('size of input:', X.shape)
            print('Size of output:',Y.shape)


    def train(self, X, Y):

        A = [np.atleast_2d(X)]                              # Init array to store layer outputs
        for layer in range(len(self.W)):                    # FP
            if self.bias_on_output:
                a = np.dot(A[layer],self.W[layer]) + self.B[layer]
            else:
                if layer == len(self.W)-1:
                    a = np.dot(A[layer],self.W[layer])
                else:
                    a = np.dot(A[layer],self.W[layer]) + self.B[layer]

            a1 = self.act_func(a)
            A.append(a1)

        self.error = A[-1] - Y

        D = [self.error*self.act_func_dif(A[-1])]

        for layer in range(len(A)-2,0,-1):              # Delta for hidden layers. Input layer is excluded
            delta = np.dot(D[-1],self.W[layer].T)*self.act_func_dif(A[layer])
            D.append(delta)
        
        # Flip D matrix
        D = D[::-1]

        for layer in range(len(self.W)):
            # self.W[layer] += -self.alpha * np.dot(A[layer].T,D[layer])
            self.W[layer] = self.W[layer] - self.alpha * np.dot(A[layer].T,D[layer])

        for layer in range(len(self.B)):
            self.B[layer] = self.B[layer] - self.alpha * D[layer]



    def predict(self,X):
        p = X
        for layer in range(len(self.W)):
            if layer == len(self.W) - 1:
                p = self.act_func(np.dot(p,self.W[layer]))
            else:
                p = self.act_func(np.dot(p,self.W[layer]) + self.B[layer])
        return p
    
    def calculate_loss(self, X, targets):           # Use this one, quadratic loss function.
		# make predictions for the input data points then compute
		# the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X)
        # loss = 0.5 * np.sum((predictions - targets) ** 2)
        loss = 1/(len(predictions)) * np.sum((predictions - targets) ** 2)
        self.loss_array.append(loss)
		# return the loss
        return loss
    
    def plot_loss(self):
        plt.figure(figsize = (10,6))
        x_axis = [0] + [(i + 1) * self.update for i in range(len(self.loss_array) - 1)]
        # plt.plot(range(0,self.epochs,self.update),self.loss_array,label = "Loss")
        plt.plot(x_axis,self.loss_array,label = "Loss")
        plt.yscale('log')
        plt.xlabel("Epochs")
        plt.ylabel("Loss (log scale)")
        plt.title(f'Training loss over epochs, every {self.update} epochs')
        plt.legend()
        plt.show


    def NNInfo(self):

        print('... Neural Network: ...')
        print('\t Network size:', self.size)
        print('\t Activation function: ', self.act)

        print('\t Weights size:')
        for i in range(len(self.W)):
            # print('\t', self.W[i].size)
            print(f'\t \t Layer {i}: {self.W[i].shape}', '(',self.W[i].size,')')

        print(f'\t Biases: (Output bias: {self.bias_on_output})')
        if self.bias_on_output:
            for i in range(len(self.size)-1):
                print(f'\t \t layer {i+1}: {self.B[i].size}')
        else:
            for i in range(len(self.size)-2):
                print(f'\t \t layer {i+1}: {self.B[i].size}')






class mnistData:
    def __init__(self,debug = False,PCA = False, PCA_num = 10):
        self.debug = debug
        self.dim = PCA
        self.pca_n = PCA_num
    def create_dataset(self,set):
        '''
        Function for creating complete training and test sets containing
        all classes.
        '''
        #Empty list
        trainset = []
        traintargets =[]
        testset = []
        testtargets =[]
        
        #For each class
        for i in range(10):
            trainset.append(set["train%d"%i])
            traintargets.append(np.full(len(set["train%d"%i]),i))
            testset.append(set["test%d"%i])
            testtargets.append(np.full(len(set["test%d"%i]),i))
        
        #Concatenate into to complete datasets
        trainset = np.concatenate(trainset)
        traintargets = np.concatenate(traintargets)
        testset = np.concatenate(testset)
        testtargets = np.concatenate(testtargets)

        if self.dim:
            PCA_n_components = self.pca_n           ## Number of dimensions.
            pca = PCA(n_components=PCA_n_components)
            train_set = pca.fit_transform(trainset)
            test_set = pca.fit_transform(testset)
            return train_set, traintargets, test_set, testtargets
        else:
            return trainset, traintargets, test_set, testtargets

    def reduced_dataset(self,set):
        '''
        Function for creating complete training and test sets containing
        all classes.
        '''
        #Empty list
        trainset = []
        traintargets =[]
        
        #For each class
        for i in range(10):
            trainset.append(set["train%d"%i])
            traintargets.append(np.full(len(set["train%d"%i]),i))
        
        #Concatenate into to complete datasets
        trainset = np.concatenate(trainset)
        traintargets = np.concatenate(traintargets)

        if self.dim:
            PCA_n_components = self.pca_n           ## Number of dimensions.
            pca = PCA(n_components=PCA_n_components)
            train_set = pca.fit_transform(trainset)
            return train_set, traintargets
        else:
            return trainset, traintargets
        
    def create_partial__dataset(self,set,num_set):
        trainset = []
        traintargets =[]
        testset = []
        testtargets =[]

        for i in num_set:
            trainset.append(set["train%d"%i])
            traintargets.append(np.full(len(set["train%d"%i]),i))
            testset.append(set["test%d"%i])
            testtargets.append(np.full(len(set["test%d"%i]),i))

        trainset = np.concatenate(trainset)
        traintargets = np.concatenate(traintargets)
        testset = np.concatenate(testset)
        testtargets = np.concatenate(testtargets)

        if self.dim:
            PCA_n_components = self.pca_n           ## Number of dimensions.
            pca = PCA(n_components=PCA_n_components)
            train_set = pca.fit_transform(trainset)
            return train_set, traintargets, testset, testtargets
        else:
            return trainset, traintargets, testset, testtargets

            