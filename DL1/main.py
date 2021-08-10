from keras.datasets import mnist
from NN import *

if __name__ == '__main__':
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    np.random.seed(10)
    use_batchnorm = True
    use_dropout = False

    input_layer = 784
    network_dims = [input_layer, 20, 7, 5, 10]

    # Normalizing the data
    X_train = X_train.reshape(X_train.shape[0], 784) / 255
    X_test = X_test.reshape(X_test.shape[0], 784) / 255
    Y_train = one_hot_10(Y_train)
    Y_test = one_hot_10(Y_test)

    # Transposing the data, to fit the L_layer_model() and Predict() format
    X_train = X_train.T
    X_test = X_test.T
    Y_train = Y_train.T
    Y_test = Y_test.T

    # Training and testing the model without batchnorm
    trained_params, costs = L_layer_model(X_train, Y_train, network_dims, learning_rate=0.009, num_iterations=15000, batch_size=50, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
    train_acc = Predict(X_train, Y_train, trained_params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)
    test_acc = Predict(X_test, Y_test, trained_params, use_batchnorm=use_batchnorm, use_dropout=use_dropout)

    details = {'Train Accuracy': train_acc, 'Test Accuracy': test_acc}
    print(f"Train Accuracy:\t {train_acc}")
    print(f"Test Accuracy:\t {test_acc}")

    # Writing the results to a file
    path = 'details.txt'
    with open(path, 'a') as f:
        print(details, file=f)
