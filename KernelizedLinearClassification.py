#OBIETTIVO : Trainare 3 modelli : Perceptron, PegasosSVM e RegularizedLogistic
import pandas as pd
import numpy as np
import itertools
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from modelss import Perceptron, PegasosSVM, RegularizedLogistic, KernelizedPerceptron, KernelizedPegasosSVM
from sklearn.model_selection import KFold
def polynomial_feature_expansion(X):
    n_samples, n_features = X.shape
    # Numero totale di feature: originali + combinazioni di grado 2
    n_poly_features = n_features + (n_features * (n_features + 1)) // 2
    
    # Creiamo una matrice per le nuove feature
    X_poly = np.zeros((n_samples, n_poly_features))
    
    # Copia le feature originali nelle prime colonne
    X_poly[:, :n_features] = X
    
    # Riempie le nuove colonne con le combinazioni quadratiche
    current_column = n_features
    for i in range(n_features):
        for j in range(i, n_features):
            X_poly[:, current_column] = X[:, i] * X[:, j]
            current_column += 1
            
    return X_poly
def grid_search(model_c, param_grid, X, y):
    keys, values = zip(*param_grid.items())
    combinations = [dict(zip(keys,v)) for v in itertools.product(*values)]
    
    best_score = -float('inf')
    best_params = None
    
    for params in combinations:
        cv_scores = []
        kf = KFold(n_splits=5, shuffle = True, random_state = 42)
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]           
            model = model_c(**params)
            model.train(X_train,y_train)
            score = model.evaluate(X_test,y_test)
            cv_scores.append(score)
        mean_score = np.mean(cv_scores)
        print(f"Tested parameter : {params}, current accuracy : {mean_score}")
        if mean_score > best_score:
            best_score = mean_score
            best_params = params
    return best_params, best_score


#Load dataset
df = pd.read_csv('dataset.csv') 

#Check for NaN Values
print("Valori mancanti per colonna:")
print(df.isnull().sum())

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values 

#Standardization : zero mean, 1 standard deviation
scaler_standard = StandardScaler()
X_scaled_standard = scaler_standard.fit_transform(X)

#Divide data in train and test. 
X_train_unscaled, X_test_unscaled, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler_standard = StandardScaler()
X_train = scaler_standard.fit_transform(X_train_unscaled)
X_test = scaler_standard.fit_transform(X_test_unscaled)

#Polynomial feature expansion
X_train_poly = polynomial_feature_expansion(X_train)
X_test_poly = polynomial_feature_expansion(X_test)



###START TRAINING###

#Perceptron algorithm execution :

#Grid search with K-Field Cross Validation for hyperparameter tuning : 
perceptron_param_grid = {
    'learning_rate' : [0.01,0.1,1],
    'epochs' : [10,50,100]
}
print("Grid search for Perceptron")
best_params, best_score = grid_search(Perceptron,perceptron_param_grid,X_train,y_train)
print(f"Best params for Perceptron: {best_params}, Best cross-validated score: {best_score}")

#Perceptron model
perc_model = Perceptron(**best_params)
perc_model.train(X_train, y_train,show_accuracy=True)

accuracy = perc_model.evaluate(X_test, y_test)
print(f"Test accuracy for Perceptron: {accuracy}")

#Perceptron with polynomial feature expansion
polyperc_model = Perceptron(**best_params)
polyperc_model.train(X_train_poly,y_train,show_accuracy=True)

accuracy = polyperc_model.evaluate(X_test_poly, y_test)
print(f"Test accuracy for Perceptron with polynomial feature expansion : {accuracy}")

#Weight comparing : 

print("Weights of perceptron :")
print(perc_model.weights)

print("Weights of perceptron with polynomial feature expansion :")
print(polyperc_model.weights)


#Kernelized Perceptron :

#With Polynomial Kernel : 
poliperc_param_grid = {
    'epochs' : [10,50,100],
    'kernel' : ['polynomial'],
    'degree' : [2,3,4],
    'coef0' : [1.0]   
}

print("Grid search for Kernelized Perceptron with polynomial kernel")
best_params, best_score = grid_search(KernelizedPerceptron,poliperc_param_grid,X_train,y_train)
print(f"Best params for Kernelized Polynomial Perceptron: {best_params}, Best cross-validated score: {best_score}")

polikern_model = KernelizedPerceptron(**best_params)
polikern_model.train(X_train,y_train,show_accuracy=True)

accuracy = polikern_model.evaluate(X_test,y_test)
print(f"Test accuracy for Kernelized polynomial Perceptron: {accuracy}")

#With Gaussian Kernel :
gaussperc_param_grid = {
    'epochs' : [50,100,200],
    'kernel' : ["gaussian"],
    'gamma' : [0.1,1,10]   
}
print("Grid search for Kernelized Perceptron with gaussian kernel")
best_params, best_score = grid_search(KernelizedPerceptron,gaussperc_param_grid,X_train,y_train)
print(f"Best params for Kernelized Gaussian Perceptron: {best_params}, Best cross-validated score: {best_score}")

gausskern_model = KernelizedPerceptron(**best_params)
gausskern_model.train(X_train,y_train,show_accuracy=True)

accuracy = gausskern_model.evaluate(X_test,y_test)
print(f"Test accuracy for Kernelized Gaussian Perceptron: {accuracy}")


#Pegasos SVM algorithm execution :
pega_param_grid = {
    'lambda_par': [0.01, 0.1, 1, 10],
    'epochs': [50,100,200],
    'batch_size': [1,32, 64, 128]
}
best_params, best_score = grid_search(PegasosSVM,pega_param_grid,X_train,y_train)
print(f"Best params for PegasosSVM: {best_params}, Best cross-validated score: {best_score}")

#Pegasos SVM Model :
SVM_model = PegasosSVM(**best_params)
SVM_model.train(X_train,y_train,show_accuracy=True)

accuracy = SVM_model.evaluate(X_test,y_test)
print(f"Test accuracy for Pegasos SVM: {accuracy}")

#Pegasos SVM With Polynomial feature expansion : 
poly_SVM_model = PegasosSVM(**best_params)
poly_SVM_model.train(X_train_poly,y_train,show_accuracy=True)

accuracy = poly_SVM_model.evaluate(X_test_poly,y_test)
print(f"Test accuracy for Pegasos SVM with polynomial feature expansion : {accuracy}")

#Regularized Logistic classification algorithm execution :
logreg_param_grid = {
    'lambda_par': [0.001, 0.01, 0.1, 1, 10],
    'epochs': [50,100,200],
    'batch_size': [32, 64, 128],
}
best_params, best_score = grid_search(RegularizedLogistic,logreg_param_grid,X_train,y_train)
print(f"Best params for PegasosSVM: {best_params}, Best cross-validated score: {best_score}")

RegLog_model = RegularizedLogistic(**best_params)
RegLog_model.train(X_train,y_train)

accuracy = RegLog_model.evaluate(X_test,y_test)
print(f"Test accuracy for Regularized Logistic Classification : {accuracy}")

poly_RegLog_model = RegularizedLogistic(**best_params)
poly_RegLog_model.train(X_train_poly,y_train)

accuracy = poly_RegLog_model.evaluate(X_test_poly,y_test)
print(f"Test accuracy for Regularized Logistic Classification with polynomial feature expansion : {accuracy}")

#Kernelized Pegasos SVM :

#Polynomial Kernel :
polipega_param_grid = {
    'lambda_par': [0.001, 0.01, 0.1, 1, 10],
    'epochs': [50,100,200],
    'batch_size': [32, 64, 128],
    'kernel' : ['polynomial'],
    'degree' : [2,3,4],
    'coef0' : [0.0,1.0]
}
best_params, best_score = grid_search(KernelizedPegasosSVM,polipega_param_grid,X_train,y_train)
print(f"Best params for PegasosSVM with polynomial kernel : {best_params}, Best cross-validated score: {best_score}")

poly_PegaSVM = KernelizedPegasosSVM(**best_params)
poly_PegaSVM.train(X_train,y_train)

accuracy = poly_PegaSVM.evaluate(X_test,y_test)
print(f"Test accuracy for Pegasos SVM with polynomial kernel : {accuracy}")
#Gaussian Kernel :
gausspega_param_grid = {
    'lambda_par': [0.001, 0.01, 0.1, 1, 10],
    'epochs': [50,100,200],
    'batch_size': [32, 64, 128],
    'kernel' : ['gaussian'],
    'gamma' : [0.1,1],
}
best_params, best_score = grid_search(KernelizedPegasosSVM,gausspega_param_grid,X_train,y_train)
print(f"Best params for PegasosSVM with gaussian kernel : {best_params}, Best cross-validated score: {best_score}")

gauss_PegaSVM = KernelizedPegasosSVM(**best_params)
gauss_PegaSVM.train(X_train,y_train)

accuracy = poly_PegaSVM.evaluate(X_test,y_test)
print(f"Test accuracy for Pegasos SVM with gaussian kernel : {accuracy}")




