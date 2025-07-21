import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt


def initializeWeights(n_in, n_out):
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    result = 1.0 / (1.0 + np.exp(-z))
    return result


def preprocess():
    mat = loadmat('mnist_all.mat') 
    train_data = []
    train_label = []
    for i in range(10):
        data = "train" + str(i)
        data = mat[data]
        train_data.append(data)
        for j in range(data.shape[0]):
            train_label.append(int(i))
    train_data = np.vstack(train_data)
    train_label = np.array(train_label)

    #random shuffle and split into train/validation
    np.random.seed(50)
    rand = np.random.permutation(len(train_data))
    train_data, train_label = train_data[rand], train_label[rand]

    validation_data = train_data[50000:]
    validation_label = train_label[50000:]
    train_data = train_data[:50000]
    train_label = train_label[:50000]


    #testing data
    test_data = []
    test_label = []
    for i in range(10):
        data = "test" + str(i)
        data = mat[data]
        test_data.append(data)
        for j in range(data.shape[0]):
            test_label.append(i)
    test_data = np.vstack(test_data)
    test_label = np.array(test_label)

    remove_index_list = [] #for redudant index
    keep_index_list = [] #the indexes chosen for training dataset

    for index in range(train_data[0].shape[0]):
        count = 0
        for vector in train_data:
            count += int(vector[index])
        if count / train_data.shape[0] == train_data[0][index]:
            remove_index_list.append(index)
        else:
            keep_index_list.append(index)
    train_data = np.delete(train_data, remove_index_list, axis=1)
    validation_data = np.delete(validation_data, remove_index_list, axis=1)
    test_data = np.delete(test_data, remove_index_list, axis=1)

    # Normalizing all data based on MNIST data set to improve accuracy
    train_data = train_data/255.0
    validation_data = validation_data/255.0
    test_data = test_data/255.0

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    biasTerms = np.ones((training_data.shape[0], 1)) 
    training_data_with_bias = np.concatenate((biasTerms, training_data), axis=1)
    z_j = sigmoid(np.dot(training_data_with_bias, w1.T)) # Calculate z values by applying activation function to a_j (the dot product of the training data with bias and the weights (units) of the hidden layer)

    z_j = np.concatenate((biasTerms,z_j),axis = 1)
    #Second transformation from hidden layer to output layer
    b_l = np.dot(z_j, w2.T) # Calculate b_l values(the dot product of the training data with bias and the weights (units) of the output layer)
    o_l = sigmoid(b_l) # Apply activation function to b_l

    # Prevents log(1) or log(0) 
    smallVal = 1e-10
    o_l = np.clip(o_l,smallVal,1-smallVal)
    # Error Function and Backpropagation section from documentation
    # Negative log likelihood error function
    # Create one-hot encoded (Similar to the binary encoding method used in class for features and K-hot encoding) labels due to poor performace when reshaping training_label like so training_label.reshape(-1,1)
    y_one_hot = np.zeros((training_data.shape[0], n_class))
    for i in range(training_data.shape[0]):
      y_one_hot[i, int(training_label[i])] = 1

    Error_Function = (np.sum(y_one_hot * np.log(o_l) + (1 - y_one_hot) * np.log(1 - o_l)))/(-training_data.shape[0]) + (lambdaval/(2*training_data.shape[0])) * (np.sum(w1[:,1:] ** 2) + np.sum(w2[:,1:] ** 2))
    # The first half of the function is the negative log likelihood error function. Where as the second term includes the regularization term.
    obj_val = Error_Function


    outputLayerError = o_l - y_one_hot #Compute the output layer error, since o_l has multiple classes we have to reshape the training_label into a 2D column vector for computation
    
    dj_dw2 = np.dot(outputLayerError.T, z_j)/training_data.shape[0] # Calculate the gradient of the error function with respect to w2
    dj_dw2[:,1:] += ((lambdaval * w2[:,1:])/training_data.shape[0]) # Regularization parameter for nonbias weights as specified in the doc
    
    hiddenLayerError = (1-z_j[:,1:]) * z_j[:,1:] * np.dot(outputLayerError,w2[:,1:]) # Here I remove the bias terms from z_j and w2 with [:,1:] essentially only referencing the original 256 nodes and calulate the error on the hidden layer nodes 
    dj_dw1 = np.dot(hiddenLayerError.T,training_data_with_bias)/training_data.shape[0]  # Calculate the gradient of the error function with respect to w1 
    dj_dw1[:,1:] += ((lambdaval*w1[:,1:])/training_data.shape[0]) # Regularization parameter for nonbias weights as specified in the doc

    # Compress gradient data into a 1D array
    vectorized_dj_dw1 = dj_dw1.flatten()
    vectorized_dj_dw2 = dj_dw2.flatten()  

    obj_grad = np.array([])
    obj_grad = np.concatenate((vectorized_dj_dw1, vectorized_dj_dw2),0)
    # Return the value of the error function and the gradient of the error function.
    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    labels = np.array([])

    biasTerms = np.ones((data.shape[0], 1))
    data_with_bias = np.concatenate((biasTerms,data),axis=1)  # add bias to input
    a_j = np.dot(data_with_bias, w1.T)
    z_j = sigmoid(a_j)

    z_j_with_bias = np.concatenate((biasTerms,z_j),axis=1)  # set hidden layer's bias to 1
    b_l = np.dot(z_j_with_bias, w2.T)
    o_l = sigmoid(b_l)

    labels = np.argmax(o_l, axis=1)

    return labels



if __name__ == "__main__":
    
        
    train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

    # set the number of nodes in input unit (not including bias unit)
    n_input = train_data.shape[1]

    # set the number of nodes in hidden unit (not including bias unit)
    n_hidden = 64

    # set the number of nodes in output unit
    n_class = 10

    # initialize the weights into some random matrices
    initial_w1 = initializeWeights(n_input, n_hidden)
    initial_w2 = initializeWeights(n_hidden, n_class)

    # unroll 2 weight matrices into single column vector
    initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

    # set the regularization hyper-parameter
    lambdaval = 0.01

    args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

    # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

    opts = {'maxiter': 50}  # Preferred value.

    nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

    # Reshape nnParams from 1D vector into w1 and w2 matrices
    w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # Test the computed parameters

    predicted_label = nnPredict(w1, w2, train_data)

    print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, validation_data)

    print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

    predicted_label = nnPredict(w1, w2, test_data)

    print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')


