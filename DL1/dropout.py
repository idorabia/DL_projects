import sys
import numpy as np
import math
from sklearn.metrics import accuracy_score


# Section 1 - Forward Propagation
def initialize_parameters(layer_dims):
    """
    Initialize parameters for the first iteration of the training phase.
    :param layer_dims:
    :return:
    """
    parameters = {}

    for i in range(1, len(layer_dims)):
        current_dim = layer_dims[i]
        prev_dim = layer_dims[i-1]

        parameters.update({f"W{i}": np.random.randn(current_dim, prev_dim) * 0.15})
        parameters.update({f"b{i}": np.zeros((current_dim, 1))})
    return parameters


def linear_forward(A, W, b):
    """
    The linear part of a layer's forward propagation.
    :param A:
    :param W:
    :param b:
    :return:
    """
    linear_cache = (A, W, b)
    Z = np.dot(W, A) + b
    # --------------------------------------- Dropout ------------------------------------------------------------
    u1 = np.random.binomial(1, 0.9, size=Z.shape) / 0.9
    Z *= u1

    # dropout_L_i = np.random.choice([0, 1], size=[next_A.shape[0], 1], p=[1-keep_prob, keep_prob])
    # a_i = np.zeros_like(next_A)
    # dropout_L_i = a_i+dropout_L_i
    # next_A = np.multiply(next_A, dropout_L_i)
    # next_A = next_A/keep_prob
    # ------------------------------------------------------------------------------------------------------------

    return Z, linear_cache


def softmax(Z):
    """
    Apply a softmax function on Z
    :param Z:
    :return:
    """
    activation_cache = Z
    exp_Z = np.exp(Z)
    sum_exp_Z = np.sum(exp_Z, axis=0)
    A = exp_Z/sum_exp_Z
    return A, activation_cache


def relu(Z):
    """
    Apply a ReLU function on Z
    :param Z:
    :return:
    """
    A = np.maximum(Z, 0)
    activation_cache = Z
    return A, activation_cache


def linear_activation_forward(A_prev, W, B, activation):
    """
    The forward propagation for the LINEAR->ACTIVATION layer
    :param A_prev:
    :param W:
    :param B:
    :param activation:
    :return:
    """
    cache = {}
    Z, linear_cache = linear_forward(A_prev, W, B)
    if activation == 'relu':
        A, activation_cache = relu(Z)
    else:
        A, activation_cache = softmax(Z)
    cache.update({'activation': activation_cache, 'linear': linear_cache})
    return A, cache


def L_model_forward(X, parameters, use_batchnorm, keep_prob=0.9):
    """
    The forward propagation for the entire network
    :param X:
    :param parameters:
    :param use_batchnorm:
    :return:
    """
    caches = []
    next_A = None
    layers_amnt = len(parameters) // 2 # floor division

    # For networks with more than one layer
    if layers_amnt > 1:
        for i in range(1, layers_amnt):
            current_W = parameters[f'W{i}']
            current_b = parameters[f'b{i}']
            if i == 1:
                next_A, prev_cache = linear_activation_forward(X, current_W, current_b, "relu")
            else:
                next_A, prev_cache = linear_activation_forward(next_A, current_W, current_b, "relu")

# --------------------------------------- Dropout ------------------------------------------------------------
#             u1 = np.random.binomial(1, keep_prob, size=next_A.shape) / keep_prob
#             next_A *= u1

            # dropout_L_i = np.random.choice([0, 1], size=[next_A.shape[0], 1], p=[1-keep_prob, keep_prob])
            # a_i = np.zeros_like(next_A)
            # dropout_L_i = a_i+dropout_L_i
            # next_A = np.multiply(next_A, dropout_L_i)
            # next_A = next_A/keep_prob
# ------------------------------------------------------------------------------------------------------------
            if use_batchnorm:
                next_A = apply_batchnorm(next_A)

            caches.append(prev_cache)
        # linear_forward of the last layer
        current_W = parameters[f'W{layers_amnt}']
        current_b = parameters[f'b{layers_amnt}']
        next_A, prev_cache = linear_activation_forward(next_A, current_W, current_b, "softmax")

    # For networks with one layer
    else:
        current_W = parameters['W1']
        current_b = parameters['b1']
        next_A, prev_cache = linear_activation_forward(X, current_W, current_b, "softmax")

    # Saving the results of the last layer
    caches.append(prev_cache)
    AL = next_A

    return AL, caches


def compute_cost(AL, Y):
    """
    Computing the cross-entropy cost between predicted labels and actual labels
    :param AL:
    :param Y:
    :return:
    """
    m = AL.shape[1]
    log_AL = np.log(AL)

    cost = (-1/m) * np.sum(Y.T * log_AL.T)

    return cost


def apply_batchnorm(A):
    """
    Normalizing each node according to its mean and variance in the current batch.
    :param A:
    :return:
    """
    A_mean = np.mean(A, axis=1, keepdims=True)
    A_var = np.var(A, axis=1, keepdims=True)
    NA = (A - A_mean) / np.sqrt(A_var + sys.float_info.epsilon)

    return NA

















def Linear_backward(dZ, cache):
    """
    The linear part of the backward propagation process for a single layer
    :param dZ:
    :param cache:
    :return:
    """
    A_prev, W, b = cache

    m = A_prev.shape[1]
    dW = np.dot(dZ, A_prev.T) / float(m)  # dividing by m (num of examples)
    db = np.sum(dZ, axis=1, keepdims=True) / float(m)  # calculating avg for each example (example = row)

    dA_prev = np.dot(W.T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    The backward propagation for the LINEAR->ACTIVATION layer
    :param dA:
    :param cache:
    :param activation:
    :return:
    """
    activation_cache = cache["activation"]

    if str.lower(activation) == "relu":
        dZ = relu_backward(dA, activation_cache)
    elif str.lower(activation) == "softmax":
        dZ = softmax_backward(dA, activation_cache)
    else:
        raise Exception("Supports only ReLU and Softmax as activation function")

    dA_prev, dW, db = Linear_backward(dZ, cache["linear"])
    return dA_prev, dW, db


def relu_backward(dA, activation_cache):
    """
    Computes the backward propagation for all ReLU nodes in a layer
    :param dA:
    :param activation_cache:
    :return:
    """
    Z = activation_cache
    dZ = np.array(dA, copy=True)
    dZ[np.where(Z <= 0)] = 0
    return dZ


def softmax_backward(dA, activation_cache):
    """
    Computes the backward propagation for all softmax nodes in a layer
    :param dA:
    :param activation_cache:
    :return:
    """
    dZ = np.array(dA, copy=True)
    return dZ


def L_model_backward(AL, Y, caches):
    """
    Computes the backward propagation for the entire network.
    :param AL:
    :param Y:
    :param caches:
    :return:
    """
    dA = AL - Y
    layers_amnt = len(caches)
    last_relu_layer = layers_amnt - 1
    grads = {}

    # handling last layer
    last_cache = caches[last_relu_layer]
    dA_prev, dW, db = linear_activation_backward(dA, last_cache, "softmax")
    grads[f'dA{last_relu_layer + 1}'] = dA
    grads[f'dW{last_relu_layer + 1}'] = dW
    grads[f'db{last_relu_layer + 1}'] = db

    # handling rest of the layers
    for i in range(last_relu_layer, 0, -1):
        curr_layer_dA = dA_prev
        curr_cache = caches[i-1]

        dA_prev, dW, db = linear_activation_backward(curr_layer_dA, curr_cache, "relu")
        grads[f'dA{i}'] = curr_layer_dA
        grads[f'dW{i}'] = dW
        grads[f'db{i}'] = db

    return grads


def Update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters of the network according to the learning rate and the gradients.
    :param parameters:
    :param grads:
    :param learning_rate:
    :return:
    """
    layers_amnt = len(parameters) // 2  # floor division
    old_params = parameters
    parameters = {}

    for i in range(layers_amnt, 0, -1):  # iterating all layers backwards
        W_l = old_params[f"W{i}"]
        W_l_grads = grads[f"dW{i}"]
        b_l = old_params[f"b{i}"]
        b_l_grads = grads[f"db{i}"]

        parameters[f"W{i}"] = W_l - (learning_rate * W_l_grads)
        parameters[f"b{i}"] = b_l - (learning_rate * b_l_grads)

    return parameters











def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False):
    """
    Train a Neural Network.
    First we initialize parameters randomly. Then we split the data to train & validation sets.
    Then we split the training-set to batches, and train the network on each batch.
    We stop the training when we reached <num_iterations> OR when the network stopped improving for <validation_intervals> iterations.
    For every 100 iterations we save the cost in a costs-list.
    Finally we save the execution details in a text file.
    :param X:
    :param Y:
    :param layers_dims:
    :param learning_rate:
    :param num_iterations:
    :param batch_size:
    :param use_batchnorm:
    :return:
    """
    cost_iterations_interval = 100
    # validation_intervals = 350
    validation_intervals = 350
    last_acc = 0.0000001
    iterations_with_no_improvement = 0
    epoch = 0
    i = 0

    params = initialize_parameters(layers_dims)
    costs = []

    X_train, X_validation, Y_train, Y_validation = _split_test_train(X, Y, 0.2)

    X_train_copy = np.array(X_train.T, copy=True)
    X_validation = X_validation.T
    Y_train_copy = np.array(Y_train.T, copy=True)
    Y_validation = Y_validation.T

    batches = _create_batches(X_train_copy, Y_train_copy, batch_size)


    while True:
        for batch_X, batch_Y in batches:

            AL, caches = L_model_forward(batch_X, params, use_batchnorm)

            #  Saving the costs
            if i % cost_iterations_interval == 0:
                costs.append(compute_cost(AL, batch_Y))

            grads = L_model_backward(AL, batch_Y, caches)

            params = Update_parameters(params, grads, learning_rate)
            i += 1

            # Validation step
            current_acc = Predict(X_validation.T, Y_validation.T, params, use_batchnorm=use_batchnorm)
            if current_acc <= last_acc:
                iterations_with_no_improvement += 1
            else:
                iterations_with_no_improvement = 0
                last_acc = current_acc

            if iterations_with_no_improvement == validation_intervals or i > num_iterations:
                _save_details_to_file(epoch, i, batch_size, use_batchnorm, costs, current_acc)
                print(f"Validation Accuracy with use_batchnorm={use_batchnorm}:\t {current_acc}")
                return params, costs

        epoch += 1

    print(f"Validation Accuracy with Batchnorm={use_batchnorm}:\t {current_acc}")
    _save_details_to_file(epoch, i, batch_size, use_batchnorm, costs, current_acc)
    return params, costs


def _save_details_to_file(epoch, iterations, batch_size, use_batchnorm, costs, validation_acc):
    """
    Saving the execution details in a file.
    :param epoch: Total epochs in the execution
    :param iterations: Total iterations in the execution
    :param batch_size: The batch_size used in the execution
    :param use_batchnorm: True/False
    :param costs: The costs list of the execution
    :param validation_acc: The accuracy of the Network on the validation-set
    :return:
    """
    details = {'Total Epochs': epoch, 'Total Iterations': iterations, 'Batch Size': batch_size,
               'Is Batchnorm': use_batchnorm, 'Validation Accuracy': validation_acc}
    path = 'batchnorm_details.txt' if use_batchnorm else 'details.txt'
    with open(path, 'a') as f:
        print(details, file=f)
        print(costs, file=f)


def Predict(X, Y, parameters, use_batchnorm=False):
    """
    Running the trained Network on new examples, and computes the accuracy.
    :param X:
    :param Y:
    :param parameters:
    :param use_batchnorm:
    :return:
    """
    Y_pred, caches = L_model_forward(X.T, parameters, use_batchnorm)
    accuracy = accuracy_score(Y.argmax(axis=1), Y_pred.T.argmax(axis=1))
    return accuracy

def _split_test_train(X, Y, test_ratio):
    """
    Given X and Y, we split them according to the given <test_ratio>
    :param X: The original set of examples
    :param Y: The original set of labels
    :param test_ratio: The required ratio for the split
    :return:
    """
    X_copy = np.array(X, copy=True)
    Y_copy = np.array(Y, copy=True)

    test_size = int(test_ratio * X_copy.shape[0])
    shuffled_indices = np.random.choice(X_copy.shape[0], test_size, replace=False)

    X_test = X_copy[shuffled_indices, :]
    Y_test = Y_copy[shuffled_indices, :]

    X_train = np.delete(X_copy, shuffled_indices, axis=0)
    Y_train = np.delete(Y_copy, shuffled_indices, axis=0)

    return X_train, X_test, Y_train, Y_test

def _create_batches(X, Y, batch_size):
    """
    Given X and Y, we split them to batches with size=<batch_size>.
    We first make a random permutation of the given data, and then do the split.
    :param X: The original set of examples
    :param Y: The original set of labels
    :param batch_size: The wanted size for each batch
    :return:
    """
    X_train_copy = np.array(X.T, copy=True)
    Y_train_copy = np.array(Y.T, copy=True)
    batches = []

    shuffled_indices = np.random.choice(X_train_copy.shape[0], X_train_copy.shape[0], replace=False)
    shuffled_X_train = X_train_copy[shuffled_indices, :]
    shuffled_Y_train = Y_train_copy[shuffled_indices, :]

    total_examples = X_train_copy.shape[0]

    full_batches_needed = math.floor(total_examples / batch_size)
    for i in range(full_batches_needed):
        batch_X = shuffled_X_train[i * batch_size : (i+1) * batch_size, :]
        batch_Y = shuffled_Y_train[i * batch_size : (i+1) * batch_size, :]
        batches.append((batch_X.T, batch_Y.T))

    # if the number of observations isn't devided by batch_size then adding the remaining observations as a batch
    if total_examples % batch_size != 0:
        batch_X = shuffled_X_train[full_batches_needed * batch_size : , :]
        batch_Y = shuffled_Y_train[full_batches_needed * batch_size : , :]
        batches.append((batch_X.T, batch_Y.T))

    return batches


from keras.datasets import mnist
from keras.utils import to_categorical

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# np.random.seed(42) with 150 ---> 0.8748 / 0.8884
# np.random.seed(42) with 120 ---> 0.8699 / 0.8848
np.random.seed(42)
# np.random.seed(13)

input_layer = 784
network_dims = [input_layer, 20, 7, 5, 10]

X_train = X_train.reshape(X_train.shape[0], 784) / 255
X_test = X_test.reshape(X_test.shape[0], 784) / 255
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)

use_batchnorm = False
trained_params, costs = L_layer_model(X_train, Y_train, network_dims, learning_rate=0.009, num_iterations=20000, batch_size=150, use_batchnorm=use_batchnorm)
train_acc = Predict(X_train, Y_train, trained_params, use_batchnorm=use_batchnorm)
test_acc = Predict(X_test, Y_test, trained_params, use_batchnorm=use_batchnorm)
details = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
print(f"Train Accuracy with use_batchnorm={use_batchnorm}:\t {train_acc}")
print(f"Test Accuracy with use_batchnorm={use_batchnorm}:\t {test_acc}")

path = 'batchnorm_details.txt' if use_batchnorm else 'details.txt'
with open(path, 'a') as f:
    print(details, file=f)


use_batchnorm = True
trained_params, costs = L_layer_model(X_train, Y_train, network_dims, learning_rate=0.009, num_iterations=20000, batch_size=150, use_batchnorm=use_batchnorm)
train_acc = Predict(X_train, Y_train, trained_params, use_batchnorm=use_batchnorm)
test_acc = Predict(X_test, Y_test, trained_params, use_batchnorm=use_batchnorm)
details = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
print(f"Train Accuracy with use_batchnorm={use_batchnorm}:\t {train_acc}")
print(f"Test Accuracy with use_batchnorm={use_batchnorm}:\t {test_acc}")

path = 'batchnorm_details.txt' if use_batchnorm else 'details.txt'
with open(path, 'a') as f:
    print(details, file=f)
