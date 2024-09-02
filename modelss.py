import numpy as np
import pandas as pd


class Perceptron:
    def __init__(self, learning_rate=0.01,epochs=10):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = 0

    def train(self,X,y,show_accuracy=False):
        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features)
        
        for epoch in range(self.epochs):
            for indx, x_i in enumerate(X):
                #calcolo etichetta
                output = np.dot(x_i, self.weights) + self.bias
                y_predicted = np.sign(output)
                
                #Aggiornamento pesi
                if y[indx]*y_predicted <= 0:
                    self.weights += self.learning_rate * y[indx] * x_i
                    self.bias += self.learning_rate * y[indx]

            if show_accuracy==True and epoch == self.epochs - 1:
                y_predicted_final = np.sign(np.dot(X, self.weights) + self.bias)
                accuracy = np.mean(y_predicted_final == y)
                print(f"Train accuracy for Perceptron: {accuracy}")
    def predict(self,X):
        return np.sign(np.dot(X,self.weights) + self.bias)
    def evaluate(self,X,y): 
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
class KernelizedPerceptron:
    def __init__(self, kernel = 'polynomial', epochs = 100, degree = 2, coef0 = 1, gamma = 0.1):
        self.epochs = epochs
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.alpha = None
        self.X_train = None
        self.y_train = None
    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'gaussian':
            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

            dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-dist_sq / (2 * self.gamma))
        else:
            raise ValueError("Kernel non valido")
    def train(self,X,y,show_accuracy=False):
        n_samples = X.shape[0]
        self.alpha = np.zeros(n_samples)
        self.X_train = X
        self.y_train = y
        K = self._kernel_matrix(X,X)
        y_evaluate = []
        for epoch in range(self.epochs):
           
            for i in range(n_samples):

                kernel_result = np.sum(self.alpha * self.y_train * K[:,i])
                y_pred = np.sign(kernel_result)
                if epoch == self.epochs - 1 :
                    y_evaluate.append(y_pred)
 
                if y_pred != y[i]:
                    self.alpha[i] += 1  
            if show_accuracy == True and epoch == self.epochs - 1 :
                accuracy = np.mean(y_evaluate == y)
                print(f"Train accuracy for Kernelized Perceptron with {self.kernel} Kernel : {accuracy}")
    def predict(self, X):
        y_pred = np.zeros(X.shape[0])
        K = self._kernel_matrix(self.X_train, X)
        for i in range(X.shape[0]):
            kernel_result = np.sum(self.alpha * self.y_train * K[:,i])
            y_pred[i]=np.sign(kernel_result)
        return y_pred
    def evaluate(self,X,y): 
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
  
class PegasosSVM:
    def __init__(self, lambda_par=1, epochs=100, batch_size=1):
        self.lambda_par = lambda_par
        self.epochs = epochs
        self.batch_size = batch_size
        self.w = None
        self.bias = 0

    def train(self, X, y,show_accuracy=False):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)

        for epoch in range(1,self.epochs + 1):
            #decay learning rate for each epoch
            lr = 1 / (self.lambda_par * epoch)
            indices = np.random.permutation(n_samples)
            X,y = X[indices],y[indices]
            #get batch indices 
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch, y_batch = X[batch_indices], y[batch_indices]
            gradient_w = np.zeros(n_features)
            gradient_b = 0
            for i in range(self.batch_size):
                xi,yi = X_batch[i],y_batch[i]
                #cumulate gradient and refresh at the end
                if yi * (np.dot(self.w,xi)+self.bias) < 1:
                    gradient_w += yi * xi
                    gradient_b += yi
            self.w = (1 - lr * self.lambda_par) * self.w + lr * gradient_w / self.batch_size
            self.bias += lr * gradient_b
            if show_accuracy == True and epoch == self.epochs :
                y_pred = np.sign(np.dot(X, self.w) + self.bias)
                accuracy = np.mean(y_pred == y)
                print(f"Total train accuracy for Pegasos SVM (without considering mini batch): {accuracy}")
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.bias)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
class KernelizedPegasosSVM:
    def __init__(self, lambda_par=1, epochs=100, kernel = 'polynomial', degree = 2, coef0 = 1, gamma = 0.1):
        self.lambda_par = lambda_par
        self.epochs = epochs
        self.kernel = kernel
        self.degree = degree
        self.coef0 = coef0
        self.gamma = gamma
        self.alpha = None
        self.X_train = None
        self.y_train = None
        
    def _kernel_matrix(self, X1, X2):
        if self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + self.coef0) ** self.degree
        elif self.kernel == 'gaussian':

            X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
            X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)

            dist_sq = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
            return np.exp(-dist_sq / (2 * self.gamma))
        else:
            raise ValueError("Kernel non valido")
        
    def train(self, X, y, show_accuracy = False):
        n_samples, n_features = X.shape
        K=self._kernel_matrix(X,X)
        self.alpha = np.zeros(n_samples)
        self.X_train = X
        self.y_train = y
        for epoch in range(1,self.epochs + 1):

            lr = 1 / (self.lambda_par * epoch)

            idx = np.random.randint(0, n_samples)
            xi,yi = X[idx],y[idx]
            kernel_result = np.sum(self.alpha * self.y_train * K[:,idx])
            if lr * yi * kernel_result < 1:
                self.alpha[idx] += 1         
    def predict(self, X):
        y_pred = []
        K=self._kernel_matrix(self.X_train,X)
        for idx in range(X.shape[0]):
            kernel_result = np.sum(self.alpha * self.y_train * K[:,idx])
            y_pred.append(np.sign(kernel_result))
        return np.array(y_pred)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy 
    
  
class RegularizedLogistic:
    def __init__(self, lambda_par=0.01, epochs=100, batch_size=1):
        self.lambda_par = lambda_par  
        self.epochs = epochs 
        self.batch_size = batch_size  
        self.weights = None  
        self.bias = 0  
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def train(self, X, y, show_accuracy = False):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)

        for epoch in range(1, self.epochs + 1):
            
            indices = np.random.permutation(n_samples)
            X, y = X[indices], y[indices]
            
            lr = 1 / (self.lambda_par * epoch)
            
            batch_indices = np.random.choice(n_samples, self.batch_size, replace=False)
            X_batch, y_batch = X[batch_indices], y[batch_indices]

            gradients_w = np.zeros(n_features)
            gradient_b = 0
            
            for i in range(self.batch_size):
                xi, yi = X_batch[i], y_batch[i]
                linear_output = np.dot(xi, self.weights) + self.bias
                y_pred = self.sigmoid(linear_output)
                
                gradient = (y_pred - yi) * xi
                gradients_w += gradient
                gradient_b += (y_pred - yi)
            
            gradients_w /= self.batch_size
            gradient_b /= self.batch_size

            self.weights -= lr * (gradients_w + self.lambda_par * self.weights)
            self.bias -= lr * gradient_b
            if epoch == self.epochs and show_accuracy == True:
                linear_output = np.dot(X, self.weights) + self.bias
                y_pred = self.sigmoid(linear_output)
                y_pred_classes = np.where(y_pred >= 0.5, 1, -1)
                accuracy = np.mean(y_pred_classes == y)
                print(f"Train accuracy during the last epoch: {accuracy}")        
    
    def predict(self, X):
        
        linear_output = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_output)
        return np.where(y_pred >= 0.5, 1, -1)
   
    def evaluate(self, X, y):
       
        y_pred = self.predict(X)
        accuracy = np.mean(y_pred == y)
        return accuracy
    
