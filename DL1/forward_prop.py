import sys
import numpy as np


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

        parameters.update({f"W{i}": np.random.randn(current_dim, prev_dim) * np.sqrt(2/prev_dim)})
        # parameters.update({f"W{i}": np.random.uniform(low=-1, high=1, size=(current_dim, prev_dim))})
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


def L_model_forward(X, parameters, use_batchnorm, use_dropout=False):
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
            if use_dropout:
                keep_proba = 0.9

                # Generating a 0/1 masking matrix, with the same shape as the current activation layer.
                dropout_L_i = np.random.choice([0, 1], size=[next_A.shape[0], 1], p=[1-keep_proba, keep_proba])
                a_i = np.zeros_like(next_A)
                dropout_L_i = a_i+dropout_L_i

                # Masking the current activation layer according to the 0/1 matrix we generated
                next_A = np.multiply(next_A, dropout_L_i)
                next_A = next_A/keep_proba
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
