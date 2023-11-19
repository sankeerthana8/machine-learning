import numpy
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import matplotlib.pyplot as plt
import time


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer

    # Output:
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return 1 / (1 + np.exp(-z))  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the
       training set
     test_data: matrix of training set. Each row of test_data contains
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples.
    # Your code here.

    size_n = 50000
    size_m = 10000
    size_c = 784
    train_preprocess = np.zeros(shape=(size_n, size_c))
    validation_preprocess = np.zeros(shape=(size_m, size_c))
    test_preprocess = np.zeros(shape=(size_m, size_c))
    train_label_preprocess = np.zeros(shape=(size_n,))
    validation_label_preprocess = np.zeros(shape=(size_m,))
    test_label_preprocess = np.zeros(shape=(size_m,))

    train_len, validation_len, test_len, train_label_len, validation_label_len = 0,0,0,0,0

    for key in mat:

        if "train" in key:
            label, tup, tup_perm, tup_len = get_key_attributes(key, mat)
            val = 1000
            # tag_len = tup_len - val

            train_preprocess[train_len:train_len + (tup_len - val)] = tup[tup_perm[val:], :]
            train_len += tup_len - val

            train_label_preprocess[train_label_len:train_label_len + (tup_len - val)] = label
            train_label_len += tup_len - val

            validation_preprocess[validation_len:validation_len + val] = tup[tup_perm[0:val], :]
            validation_len += val

            validation_label_preprocess[validation_label_len:validation_label_len + val] = label
            validation_label_len += val


        else:
            if "test" in key:
                label, tup, tup_perm, tup_len = get_key_attributes(key,mat)
                if tup_len is not None:
                    test_label_preprocess[test_len:test_len + tup_len] = label
                    t_perm = tup[tup_perm]
                    test_preprocess[test_len:test_len + tup_len] = t_perm
                    test_len += tup_len

    train_size, train_perm, train_data, train_label = get_data(train_preprocess, train_label_preprocess)

    validation_size, vali_perm, validation_data, validation_label = get_data(validation_preprocess, validation_label_preprocess)

    test_size, test_perm, test_data, test_label = get_data(test_preprocess, test_label_preprocess)

    # Feature selection
    # Your code here.
    total_data = np.array(np.vstack((train_data, validation_data, test_data)))
    duplicates = np.all(total_data == total_data[0, :], axis=0)

    total_data = total_data[:, ~duplicates]


    train_data = get_final_array(total_data,0,len(train_data))

    validation_start = len(train_data)
    validation_end = len(train_data) + len(validation_data)
    validation_data = get_final_array(total_data,validation_start,validation_end)

    test_start  = validation_end
    test_end = len(train_data) + len(validation_data) + len(test_data)
    test_data = get_final_array(total_data, test_start, test_end)

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label

def get_key_attributes(key, mat):
    if key is not None and mat is not None:
        t_key = mat.get(key)
        return key[-1], mat.get(key), np.random.permutation(range(t_key.shape[0])), len(t_key)

def get_final_array(final_list_value, start, end):
    return  final_list_value[start:end,:]

def get_data(preprocess, preprocess_label):
    preprocess_shape = preprocess.shape[0]
    data_perm = np.random.permutation(range(preprocess_shape))
    final_data = preprocess[data_perm]
    final_data = np.double(final_data) / 255.0
    return  range(preprocess_shape),data_perm, final_data, preprocess_label[data_perm]


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log
    %   likelihood error function with regularization) given the parameters
    %   of Neural Networks, thetraining data, their corresponding training
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    t_shape  = training_label.shape[0]
    temp_training_label = np.zeros((t_shape, 10))

    temp_training_label[np.arange(t_shape, dtype="int"), training_label.astype(int)] = 1
    training_label = temp_training_label

    array_training_data = np.array(training_data)
    td_shape = training_data.shape[0]
    training_data = np.column_stack((array_training_data, np.array(np.ones(td_shape))))

    matmul_td_tw1 = np.matmul(training_data, np.transpose(w1))
    sigma1 = sigmoid(matmul_td_tw1)
    sigma1_shape = sigma1.shape[0]
    sigma1 = np.column_stack((sigma1,  np.ones(sigma1_shape)))

    matmul_td_tw2 = np.matmul(sigma1, np.transpose(w2))
    sigma2 = sigmoid(matmul_td_tw2)

    myDelta = sigma2 - training_label

    gradW2 = np.matmul(np.transpose(myDelta), sigma1)
    gradW1 = np.delete(np.matmul(np.transpose(((1 - sigma1) * sigma1 * (np.matmul(myDelta, w2)))), training_data), n_hidden, 0)

    N = training_data.shape[0]

    lamb_2n = (lambdaval / (2 * N))
    sumOfSquares1 = np.sum(w1**2)
    sumOfSquares2 = np.sum(w2**2)

    log_sigma_2 = np.log(1 - sigma2)
    obj_val = ((np.sum(-1 * (training_label * np.log(sigma2) + (1 - training_label) * log_sigma_2))) / N) + (
            lamb_2n * (sumOfSquares1 + sumOfSquares2))

    gradW1 = (gradW1 + (lambdaval * w1)) / N
    gradW2 = (gradW2 + (lambdaval * w2)) / N

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.concatenate((gradW1.flatten(), gradW2.flatten()), 0)

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature
    %       vector of a particular image

    % Output:
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    lenData = len(data)

    data = np.column_stack([data, np.ones(lenData)])

    sigma1 = np.column_stack([sigmoid(data.dot(np.transpose(w1))), np.ones(lenData)])
    sigma2 = sigmoid(sigma1.dot(np.transpose(w2)))

    labels = np.argmax(sigma2, axis=1)

    return labels


"""**************Neural Network Script Starts here********************************"""

import csv
header = ['Lambda', 'Hidden neuron', 'Training_set', 'Validation_set', 'Testing_set', 'Time taken']

lamba = [0,5,10,15,20,30,40,50,60]
hidden_neurons = [4,8,12,16,20,30,40,50,65,80]
#lamba = [0]
#hidden_neurons = [4,8]
main_data = []


for i in range(0,len(lamba)):
    for j in range(0,len(hidden_neurons)):
        startTime = time.time()
        train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

        #  Train Neural Network

        # set the number of nodes in input unit (not including bias unit)
        n_input = train_data.shape[1]

        # set the number of nodes in hidden unit (not including bias unit)
        n_hidden = hidden_neurons[j]

        # set the number of nodes in output unit
        n_class = 10

        # initialize the weights into some random matrices
        initial_w1 = initializeWeights(n_input, n_hidden)
        initial_w2 = initializeWeights(n_hidden, n_class)

        # unroll 2 weight matrices into single column vector
        initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

        # set the regularization hyper-parameter
        lambdaval = lamba[i]

        args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

        # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

        opts = {'maxiter': 50}  # Preferred value.

        nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

        # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
        # and nnObjGradient. Check documentation for this function before you proceed.
        # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)

        # Reshape nnParams from 1D vector into w1 and w2 matrices
        w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
        w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

        # Test the computed parameters

        predicted_label = nnPredict(w1, w2, train_data)

        # find the accuracy on Training Dataset

        training_set_accuracy_value = str(100 * np.mean((predicted_label == train_label).astype(float)))
        print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

        predicted_label = nnPredict(w1, w2, validation_data)

        # find the accuracy on Validation Dataset
        validation_set_accuracy_value = str(100 * np.mean((predicted_label == validation_label).astype(float)))
        print('\n Validation set Accuracy:' + str(
            100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

        predicted_label = nnPredict(w1, w2, test_data)

        # find the accuracy on Validation Dataset
        test_set_accuracy_value = str(100 * np.mean((predicted_label == test_label).astype(float)))
        print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
        timeDifference = time.time() - startTime
        print('\n Time Taken: ' + str(timeDifference) + ' seconds.')

        data =[str(lambdaval), str(n_hidden), training_set_accuracy_value, validation_set_accuracy_value, test_set_accuracy_value ,str(timeDifference)]
        main_data.append(data)



with open('optimal_values.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(main_data)



# startTime = time.time()
# train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#
# #  Train Neural Network
#
# # set the number of nodes in input unit (not including bias unit)
# n_input = train_data.shape[1]
#
# # set the number of nodes in hidden unit (not including bias unit)
# n_hidden = 30
#
# # set the number of nodes in output unit
# n_class = 10
#
# # initialize the weights into some random matrices
# initial_w1 = initializeWeights(n_input, n_hidden)
# initial_w2 = initializeWeights(n_hidden, n_class)
#
# # unroll 2 weight matrices into single column vector
# initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)
#
# # set the regularization hyper-parameter
# lambdaval = 0
#
# args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)
#
# # Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
#
# opts = {'maxiter': 50}  # Preferred value.
#
# nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
#
# # In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# # and nnObjGradient. Check documentation for this function before you proceed.
# # nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)
#
#
# # Reshape nnParams from 1D vector into w1 and w2 matrices
# w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
# w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
#
# # Test the computed parameters
#
# predicted_label = nnPredict(w1, w2, train_data)
#
# # find the accuracy on Training Dataset
#
# print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')
#
# predicted_label = nnPredict(w1, w2, validation_data)
#
# # find the accuracy on Validation Dataset
#
# print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')
#
# predicted_label = nnPredict(w1, w2, test_data)
#
# # find the accuracy on Validation Dataset
#
# print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')
# timeDifference = time.time()-startTime
# print('\n Time Taken: ' + str(timeDifference)+ ' seconds.')

# y_axis = [64.71,86.57,91.72,92.70,93.21,94.95,95.14,95.21,95.45,94.98]
# x_axis = [4,8,12,16,20,30,40,50,64,80]
#
# plt.plot(x_axis, y_axis)
# plt.title('Hidden neurons vs Accuracy')
# plt.xlabel('Hidden neurons')
# plt.ylabel('Accuracy')
# plt.show()



