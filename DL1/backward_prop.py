import numpy as np


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

