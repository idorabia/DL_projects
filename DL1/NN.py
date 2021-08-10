import math
from forward_prop import *
from backward_prop import *


def L_layer_model(X, Y, layers_dims, learning_rate, num_iterations, batch_size, use_batchnorm=False, use_dropout=False):
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
    # Defining constants and counters
    cost_iterations_interval = 100
    validation_intervals = 350
    last_acc = 0.0000001
    iterations_with_no_improvement = 0
    epoch = 0
    i = 0
    X = X.T
    Y = Y.T
    params = initialize_parameters(layers_dims)
    costs = {}

    # Splitting the data-set to Train-Validation sets
    X_train, X_validation, Y_train, Y_validation = _split_test_train(X, Y, 0.2)
    X_train_copy = np.array(X_train.T, copy=True)
    X_validation = X_validation.T
    Y_train_copy = np.array(Y_train.T, copy=True)
    Y_validation = Y_validation.T

    # Creating batches from the training-set
    batches = _create_batches(X_train_copy, Y_train_copy, batch_size)

    while True:
        for batch_X, batch_Y in batches:

            # Forward propagation
            AL, caches = L_model_forward(batch_X, params, use_batchnorm, use_dropout)

            # Saving the costs
            if i % cost_iterations_interval == 0:
                costs[i] = (compute_cost(AL, batch_Y))
                
            # Computing the gradients with backward propagation
            grads = L_model_backward(AL, batch_Y, caches)

            # Updating the network parameters
            params = Update_parameters(params, grads, learning_rate)
            i += 1

            # Validation step
            current_acc = Predict(X_validation, Y_validation, params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
            if current_acc < last_acc:
                iterations_with_no_improvement += 1
            else:
                iterations_with_no_improvement = 0
                last_acc = current_acc

            # Stop condition
            if iterations_with_no_improvement == validation_intervals or i > num_iterations:
                _save_details_to_file(epoch, i, batch_size, use_batchnorm, costs, current_acc, use_dropout)
                print(f"Validation Accuracy with use_batchnorm={use_batchnorm}:\t {current_acc}")
                return params, costs

        epoch += 1


def Predict(X, Y, parameters, use_batchnorm=False, use_dropout=False):
    """
    Running the trained Network on new examples, and computes the accuracy.
    :param X:
    :param Y:
    :param parameters:
    :param use_batchnorm:
    :return:
    """
    Y_pred, caches = L_model_forward(X, parameters, use_batchnorm, use_dropout)
    pred_argmax = Y_pred.argmax(axis=0)
    actual_argmax = Y.argmax(axis=0)
    hit = np.count_nonzero(actual_argmax == pred_argmax)
    accuracy = hit / pred_argmax.shape[0]
    return accuracy


def _save_details_to_file(epoch, iterations, batch_size, use_batchnorm, costs, validation_acc, use_dropout):
    """
    Saving the execution details in a file.
    :param epoch: Total epochs in the execution
    :param iterations: Total iterations in the execution
    :param batch_size: The batch_size used in the execution
    :param use_batchnorm: True/False
    :param costs: The costs list of the execution
    :param validation_acc: The accuracy of the Network on the validation-set
    :param use_dropout: True/False
    :return:
    """
    details = {'Total Epochs': epoch, 'Total Iterations': iterations, 'Batch Size': batch_size,
               'Is Batchnorm': use_batchnorm, 'Validation Accuracy': validation_acc, 'Is Dropout': use_dropout}
    path = 'details.txt'
    with open(path, 'a') as f:
        print(details, file=f)
        print(costs, file=f)


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

def one_hot_10(vec):
    """
    Make a vector of labels of size (N,1) into a One Hot Matrix of size (N, 10).
    :param vec:
    :return:
    """
    size = vec.shape[0]
    one_hot_vec = np.zeros((size, 10))
    one_hot_vec[np.arange(size), vec] = 1
    return one_hot_vec


