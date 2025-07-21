'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time as time
import pickle
from scipy.optimize import minimize
from math import sqrt
import itertools
import time

# Do not change this
def initializeWeights(n_in,n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    #np.random.seed(42)
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon
    return W

def sigmoid(z):
    return 1 / (1 + np.exp(-np.array(z)))

def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    biasTerms = np.ones((training_data.shape[0], 1)) # Create bias terms for data
    training_data_with_bias = np.concatenate((biasTerms, training_data), axis=1) # Add bias terms to data
    z_j = sigmoid(np.dot(training_data_with_bias, w1.T)) # Calculate z values by applying activation function to a_j (the dot product of the training data with bias and the weights (units) of the hidden layer)

    z_j = np.concatenate((biasTerms,z_j),axis = 1)
    #Second transformation from hidden layer to output layer
    b_l = np.dot(z_j, w2.T) # Calculate b_l values(the dot product of the training data with bias and the weights (units) of the output layer)
    o_l = sigmoid(b_l) # Apply activation function to b_l

    smallVal = 1e-16
    o_l = np.clip(o_l,smallVal,1-smallVal)
    # Error Function and Backpropagation section from documentation
    # Negative log likelihood error function
    # Create one-hot encoded labels due to poor performace when reshaping training_label like so training_label.reshape(-1,1)
    y_one_hot = np.zeros((training_data.shape[0], n_class))
    for i in range(training_data.shape[0]):
      y_one_hot[i, int(training_label[i])] = 1

    Error_Function = (np.sum(y_one_hot * np.log(o_l) + (1 - y_one_hot) * np.log(1 - o_l)))/(-training_data.shape[0]) + (lambdaval/(2*training_data.shape[0])) * (np.sum(w1[:,1:] ** 2) + np.sum(w2[:,1:] ** 2))
    # The first half of the function is the negative log likelihood error function. Where as the second term includes the regularization term.
    obj_val = Error_Function


    outputLayerError = o_l - y_one_hot #Compute the output layer error, since o_l has multiple classes we have to reshape the training_label into a 2D column vector for computation
    
    dj_dw2 = np.dot(outputLayerError.T, z_j)/training_data.shape[0] # Calculate the gradient of the error function with respect to w2
    dj_dw2[:,1:] += ((lambdaval * w2[:,1:])/training_data.shape[0]) # Regularization parameter for nonbias weights
    
    hiddenLayerError = (1-z_j[:,1:]) * z_j[:,1:] * np.dot(outputLayerError,w2[:,1:]) # Here I remove the bias terms from z_h and w2 with [:,1:] essentially only referencing the original 256 nodes and calulate the error on the hidden layer nodes 
    dj_dw1 = np.dot(hiddenLayerError.T,training_data_with_bias)/training_data.shape[0]  # Calculate the gradient of the error function with respect to w1 
    dj_dw1[:,1:] += ((lambdaval*w1[:,1:])/training_data.shape[0]) # Regularization parameter for nonbias weights

    # Compress gradient data into a 1D array
    vectorized_dj_dw1 = dj_dw1.flatten()
    vectorized_dj_dw2 = dj_dw2.flatten()  

    obj_grad = np.array([])
    obj_grad = np.concatenate((vectorized_dj_dw1, vectorized_dj_dw2),0)

    return (obj_val, obj_grad)
    

def nnPredict(w1,w2,data):

    biasTerms = np.ones((data.shape[0], 1))
    data_with_bias = np.concatenate((biasTerms,data),axis=1)  # add bias to input
    a_j = np.dot(data_with_bias, w1.T)
    z_j = sigmoid(a_j)

    z_j_with_bias = np.concatenate((biasTerms,z_j),axis=1)  # set hidden layer's bias to 1
    b_l = np.dot(z_j_with_bias, w2.T)
    o_l = sigmoid(b_l)

    labels = np.argmax(o_l, axis=1)

    return labels

def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y


train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 128
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 30
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. 
opts = {'maxiter' :50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
#show hyperparameter
print("\nlabdaval= "+str(lambdaval)+"\nhidden_unit=  "+str(n_hidden))


'''#diagram generater function
def plot_hyperparameter_grid(results_df, fixed_maxiter=50):
    # Filter and prepare data
    df = results_df[results_df['maxiter'] == fixed_maxiter]
    df = df.sort_values(by=['n_hidden', 'lambda'])

    # Create pivot tables
    acc_pivot = df.pivot(index="n_hidden", columns="lambda", values="accuracy")
    time_pivot = df.pivot(index="n_hidden", columns="lambda", values="train_time")

    # Create figure
    plt.figure(figsize=(15, 10))
    ax = sns.heatmap(acc_pivot, annot=False, fmt=".1f", cmap="YlGnBu",
                     cbar_kws={'label': 'Accuracy (%)'}, vmin=0, vmax=100)

    # Add text annotations with accuracy and time
    for i in range(len(acc_pivot.index)):
        for j in range(len(acc_pivot.columns)):
            acc = acc_pivot.iloc[i, j]
            t = time_pivot.iloc[i, j]
            text = f"{acc:.1f}%\n({t:.0f}s)"
            ax.text(j + 0.5, i + 0.5, text,
                    ha='center', va='center',
                    color='white' if acc > 50 else 'black')

    # Formatting
    plt.title(f"Hyperparameter Performance Matrix")
    plt.xlabel("Lambda (Regularization Strength)")
    plt.ylabel("Hidden Units")
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)

    # Add legend for time
    ax.text(1.02, 0.3, "Cell Format:\nAccuracy %\n(Time seconds)",
            transform=ax.transAxes, fontsize=10)

    plt.tight_layout()
    plt.show()


#auto hyperparameter testing function
def hyperparameter_tuning(n_hidden_options, lambdaval_options, maxiter_options):
    all_results = []#for generate diagram
    totol_time = 0#for tracking time

    # Load and preprocess data once
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
    n_input = train_data.shape[1]
    n_class = 2

    #tracking the best accuracy and its hyperparameter
    best_accuracy = -1
    best_params = {}

    #generate all combinations
    all_combinations = list(itertools.product(n_hidden_options,
                                              lambdaval_options,
                                              maxiter_options))

    print(f"Testing {len(all_combinations)} combinations:")

    for i, (n_hidden, lambdaval, maxiter) in enumerate(all_combinations):
        print(f"\n combinaton:{i + 1}/{len(all_combinations)} ")
        print(f"n_hidden: {n_hidden}, lambda: {lambdaval}, maxiter: {maxiter}")

        #original code: initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden);
        initial_w2 = initializeWeights(n_hidden, n_class);
        #original code: unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        #original code: Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
        opts = {'maxiter': maxiter}

        train_start = time.time() #track time for each training combination

        #original code:
        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        train_time = time.time() - train_start #time tracking

        #original code:
        params = nn_params.get('x')
        #original code: Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        #original code:
        predicted_label = nnPredict(w1, w2, train_data)
        # find the accuracy on Training Dataset
        print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
        predicted_label = nnPredict(w1, w2, validation_data)
        # find the accuracy on Validation Dataset
        print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
        predicted_label = nnPredict(w1, w2, test_data)
        # find the accuracy on Validation Dataset
        print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
        print('\n training time: ' + str(train_time) + 's')

        accuracy = 100 * np.mean((predicted_label == test_label).astype(float))  # used for get highest accuracy
        totol_time = totol_time + train_time #total time used for all combination

        #used for generating diagram:
        all_results.append({
            'n_hidden': n_hidden,
            'lambda': lambdaval,
            'maxiter': maxiter,
            'accuracy': accuracy,
            'train_time': train_time
        })

        # Update best combination
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = {
                'n_hidden': n_hidden,
                'lambda': lambdaval,
                'maxiter': maxiter,
                'weights': (w1, w2),
                'accuracy': accuracy,
                'train_time': train_time
            }

    results_df = pd.DataFrame(all_results)#for generating diagram

    # Final report
    print("\n----------- Best combination ----------")
    print(f"Hidden units: {best_params['n_hidden']}")
    print(f"Lambda: {best_params['lambda']}")
    print(f"Iterations: {best_params['maxiter']}")
    print(f"Test accuracy: {best_params['accuracy']:}%")
    print(f'total time used: {totol_time}')
    return results_df #used to generate diagram

n_hidden_options = [4,8,12,16,20,32,64,128,256]
lambdaval_options = [0.001,0.01,0.1,1,5,10,15,20,25,30,35,40,45,50,55,60]#[0.0001,0.001,0.01,0.1,1]
#lambdaval_options = [0]
maxiter_options = [50]


results_df = hyperparameter_tuning(n_hidden_options, lambdaval_options, maxiter_options)
plot_hyperparameter_grid(results_df)''' 