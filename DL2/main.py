import datetime
import math
import os
import random
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomContrast, RandomZoom
from tensorflow.keras.regularizers import l2
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import BinaryAccuracy, AUC, Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import csv

GENERAL_LOC = './lfw2'
TEST_SET_PATH = './pairsDevTest.txt'
TRAIN_SET_PATH = './pairsDevTrain.txt'
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
np.random.seed(14)
tf.random.set_seed(14)

def load_image(name, img_num):
    """
    Opens a .jpg image as a numpy array
    :param name: name of the image file
    :param img_num: number of the image
    :return:
    """
    image_folder = name
    image_name = f"{name}_{int(img_num):04d}"
    img = cv2.imread(f'{GENERAL_LOC}/{image_folder}/{image_name}.jpg', 0)
    img = cv2.resize(img, dsize=(105, 105), interpolation=cv2.INTER_LINEAR)
    return img.reshape(img.shape[0],img.shape[1], 1)

def load_pair(row):
    """
    Given a row from the test/train list, and based on the row length (3 or 4 elements), generates a pair of
    (anchor,target) and returns it.
    :param row: a row from the train/test lists
    :return:
    """
    if len(row) == 3:  # Positive twins
        name = row[0]
        anchor_num = row[1]
        target_num = row[2]

        anchor = load_image(name, anchor_num)
        target = load_image(name, target_num)
        is_positive = 1

    else:  # Negative twins
        anchor_name = row[0]
        anchor_num = row[1]

        target_name = row[2]
        target_num = row[3]

        anchor = load_image(anchor_name, anchor_num)
        target = load_image(target_name, target_num)
        is_positive = 0

    return anchor, target, is_positive

def display_anchor_target(anchor, target, title=None, usage='anchor_target'):
    """
    Display the grayscale images of the given anchor and target
    :param anchor:
    :param target:
    :return:
    """
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(anchor, cmap='gray')
    ax[1].imshow(target, cmap='gray')
    if usage=='Augmantation':
        ax[0].set_title('Befor')
        ax[1].set_title('After')
    else:
        ax[0].set_title('Anchor')
        ax[1].set_title('Target')
    if title:
        plt.suptitle(title)
    plt.show()
    plt.close()

def preprocessing(train_anchors, train_target, train_positivities, validation_anchors, validation_targets, test_anchors, test_targets,
                  Augmatation=False, Normalization=False, wise='featurewise'):
    """
    Display preprocessing mechanism. The user can choose which preprocess handling he wants: Augmatation, Normalization
    or both.

    The normalization is applied on all data-sets, and performs the following:
    1. Centering – subtracting the featurewise mean from the dataset
    2. Standardizing – dividing the dataset by the featurewise standard deviation.

    The augmentation is applied only on the train-set and performs the following:
    1. RandomFlip ("horizontal_and_vertical")
    2. RandomRotation (factor =between 0.2 and 0.4)
    3. RandomZoom (factor =between -0.3 and 0.3)
    4. RandomContrast (factor =between 0.1 and0.5).
    """
    if Normalization:
        if wise == 'samplewise':
            datagen = ImageDataGenerator(samplewise_center=True, samplewise_std_normalization=True)
        else:
            datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)

        # Normalize the training dataset according to the desired normalization:
        datagen.fit(train_anchors)
        iterator_anc = datagen.flow(train_anchors, batch_size=len(train_anchors))
        train_anchors = iterator_anc.next()
        datagen.fit(train_target)
        iterator_tar = datagen.flow(train_target, batch_size=len(train_anchors))
        train_target = iterator_tar.next()

        # Normalize the validation dataset according to the desired normalization:
        datagen.fit(validation_anchors)
        iterator_anc = datagen.flow(validation_anchors, batch_size=len(validation_anchors))
        validation_anchors = iterator_anc.next()
        datagen.fit(validation_targets)
        iterator_tar = datagen.flow(validation_targets, batch_size=len(validation_targets))
        validation_targets = iterator_tar.next()

        # Normalize the test dataset according to the desired normalization:
        datagen.fit(test_anchors)
        iterator_anc = datagen.flow(test_anchors, batch_size=len(test_anchors))
        test_anchors = iterator_anc.next()
        datagen.fit(test_targets)
        iterator_tar = datagen.flow(test_targets, batch_size=len(test_targets))
        test_targets = iterator_tar.next()

    if Augmatation:
        # Including the original training dataset on the augmented dataset:
        augmented_anc = [train_anchors]
        augmented_tar = [train_target]

        # Creating an augmented dataset by flipping the samples of the training dataset, and adding the new samples to
        # the entire augmented dataset:
        data_augmentation = tf.keras.Sequential([RandomFlip("horizontal_and_vertical")])
        pre_anchors = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_anchors))
        flip_anchors = np.array(pre_anchors).reshape((1760, 105, 105, 1))
        pre_target = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_target))
        flip_target = np.array(pre_target).reshape((1760, 105, 105, 1))
        augmented_anc.append(flip_anchors)
        augmented_tar.append(flip_target)

        # Creating an augmented dataset by rotating the samples of the training dataset, and adding the new samples to
        # the entire augmented dataset:
        data_augmentation = tf.keras.Sequential([RandomRotation(factor=(0.2, 0.4))])
        pre_anchors = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_anchors))
        rotation_anchors = np.array(pre_anchors).reshape((1760, 105, 105, 1))
        pre_target = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_target))
        rotation_target = np.array(pre_target).reshape((1760, 105, 105, 1))
        augmented_anc.append(rotation_anchors)
        augmented_tar.append(rotation_target)

        # Creating an augmented dataset by changing the zoom of the samples of the training dataset, and adding the new
        # samples to the entire augmented dataset:
        data_augmentation = tf.keras.Sequential([RandomZoom((-0.3, 0.3))])
        pre_anchors = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_anchors))
        zoom_anchors = np.array(pre_anchors).reshape((1760, 105, 105, 1))
        pre_target = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_target))
        zoom_target = np.array(pre_target).reshape((1760, 105, 105, 1))
        augmented_anc.append(zoom_anchors)
        augmented_tar.append(zoom_target)

        # Creating an augmented dataset by changing the contrast of the samples of the training dataset, and adding the
        # new samples to the entire augmented dataset:
        data_augmentation = tf.keras.Sequential([RandomContrast(factor=(0.1, 0.5))])
        pre_anchors = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_anchors))
        contrast_anchors = np.array(pre_anchors).reshape((1760, 105, 105, 1))
        pre_target = list(map(lambda x: (data_augmentation(np.reshape(x, (1, 105, 105, 1)), training=True)), train_target))
        contrast_target = np.array(pre_target).reshape((1760, 105, 105, 1))
        augmented_anc.append(contrast_anchors)
        augmented_tar.append(contrast_target)

        # Adapting the shapes of the augmented dataset and the output array of the dataset:
        train_anchors = np.array(augmented_anc).reshape((8800, 105, 105, 1))
        train_target = np.array(augmented_tar).reshape((8800, 105, 105, 1))
        train_positivities = np.array([train_positivities, train_positivities, train_positivities, train_positivities,
                                       train_positivities])
        train_positivities = train_positivities.reshape(8800, 1)

    return train_anchors, train_target, train_positivities, validation_anchors, validation_targets, test_anchors, test_targets


def display_one_shot_task(task, fig_name):
    """
    Display the grayscale images of the given one-shot task
    :param task:
    :return:
    """
    anchor, pos_target, _ = task[0][0]
    pos_proba = task[1][0]

    fig, ax = plt.subplots(1, 1)
    ax.imshow(anchor, cmap='gray')
    ax.set_title('Anchor')
    plt.savefig(f'./plots/{fig_name}_anchor')
    plt.close()

    fig, ax = plt.subplots(2, 5)
    fig.set_figheight(12)
    fig.set_figwidth(12)
    ax[0][0].imshow(pos_target, cmap='gray')
    ax[0][0].set_title(f'Proba={pos_proba}', fontsize=10)
    # Iterating through negative examples
    for i in range(1, len(task[0])):
        _, curr_neg_target, _ = task[0][i]
        curr_neg_proba = task[1][i]
        ax[int(i / 5)][i % 5].imshow(curr_neg_target, cmap='gray')
        ax[int(i / 5)][i % 5].set_title(f'Proba={np.round(curr_neg_proba, 6)}', fontsize=10)
    fig.tight_layout()
    plt.savefig(f'./plots/{fig_name}_targets')
    plt.close()


def build_siamese_model(input_shape, conv_regularizer, fc_regularizer, dropout=False, dense_size=4096):
    """
    Building a Siamese NN model.
    :param input_shape:
    :return:
    """
    anchor_input = Input(input_shape)
    target_input = Input(input_shape)
    model = Sequential()

    # first layer of CONV => RELU => MAXPOOL
    model.add(Conv2D(64, (10, 10), activation="relu", kernel_regularizer=l2(conv_regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # second layer of CONV => RELU => MAXPOOL
    model.add(Conv2D(128, (7, 7), activation="relu", kernel_regularizer=l2(conv_regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # third layer of CONV => RELU => MAXPOOL
    model.add(Conv2D(128, (4, 4), activation="relu", kernel_regularizer=l2(conv_regularizer)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # fourth layer of CONV => RELU => Flatten
    model.add(Conv2D(256, (4, 4), activation="relu", kernel_regularizer=l2(conv_regularizer)))
    model.add(Flatten())

    # fifth layer - dense
    if dropout == True:
        model.add(Dense(dense_size, activation='sigmoid'))
        model.add(Dropout(0.1))
    else:
        model.add(Dense(dense_size, activation='sigmoid', kernel_regularizer=l2(fc_regularizer)))

    # embeddings distance
    anchor_embedding = model(anchor_input)
    target_embedding = model(target_input)
    embedding_dist = Lambda(lambda embeddings: K.abs(embeddings[1] - embeddings[0]))([anchor_embedding, target_embedding])

    # Last layer
    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(fc_regularizer))(embedding_dist)
    network = Model(inputs=[anchor_input, target_input], outputs=output)
    return network


def make_norm_anchor_target_set(path):
    """
    Given the path to the pairs.txt files, parse the file and generates three lists:
    1. list of anchor images
    2. list of target images
    3. list of positivities of the images (1 or 0)
    :param path:
    :return:
    """
    with open(path, "r") as f:
        lines = f.readlines()
        test_rows = list(map(lambda line: line.split('\t'), lines))[1:]

    # Creating Test set
    anchors = []
    targets = []
    is_positive = []
    for row in test_rows:
        anchor, target, positivity = load_pair(row)

        # Normalizing the images
        anchors.append(anchor/255.0)
        targets.append(target/255.0)
        is_positive.append(positivity)
    return np.array(anchors), np.array(targets), np.array(is_positive)


def decay_schedule(epoch, learning_rate):
    """
    Decay learning_rate by 1% every epoch
    :param epoch:
    :param learning_rate:
    :return:
    """
    if (epoch % 1 == 0) and (epoch != 0):
        learning_rate = learning_rate * 0.99
    return learning_rate


def _shuffle_split_train_validation(train_anchors, train_targets, train_positivities, ratio):
    """
    Split given train-set of pairs of images into train/validation sets.
    NOTE: this function splits the dataset so the resulting sets will contain equal number of SHUFFLED positive and negative examples.
    """
    train_before_split_len = len(train_anchors)
    validation_len = round(ratio * train_before_split_len)

    positive_indices = np.array(np.where(train_positivities == 1)).reshape(-1)
    negative_indices = np.array(np.where(train_positivities == 0)).reshape(-1)

    # Randomly choosing equal number of positive and negative examples to the validation set
    validation_indices = np.append(np.random.choice(positive_indices, int(validation_len/2), replace=False),
                                   np.random.choice(negative_indices, int(validation_len/2), replace=False))

    validation_anchors = train_anchors[validation_indices, :]
    validation_targets = train_targets[validation_indices, :]
    validation_positivities = train_positivities[validation_indices]
    train_anchors = np.delete(train_anchors, validation_indices, axis=0)
    train_targets = np.delete(train_targets, validation_indices, axis=0)
    train_positivities = np.delete(train_positivities, validation_indices, axis=0)

    # Shuffling train and validation sets
    train_indices = [i for i in range(len(train_anchors))]
    validation_indices = [i for i in range(len(validation_anchors))]
    random.shuffle(train_indices)
    random.shuffle(validation_indices)

    return train_anchors[train_indices], train_targets[train_indices], train_positivities[train_indices], \
           validation_anchors[validation_indices], validation_targets[validation_indices], validation_positivities[validation_indices]


def _make_one_shot_tasks(path, negatives_per_task):
    """
    Given path for test-set text file, parsing it and making a list of one-shot classification tasks.
    Each one-shot task is consisted of 1 positive pair and <negatives_per_task> negative pairs.
    """
    with open(path, "r") as f:
        lines = f.readlines()
        test_rows = list(map(lambda line: line.split('\t'), lines))[1:]

    positive_test_rows = [(row[0], row[1], row[2]) for row in test_rows if len(row) == 3]
    tasks = []

    for anchor_name, anchor_img_number, target_img_number in positive_test_rows:
        task = []

        # Loading anchor image for the task
        anchor_img = load_image(anchor_name, anchor_img_number)

        # Loading POSITIVE target image for the task
        positive_img = load_image(anchor_name, target_img_number)
        task.append((anchor_img, positive_img, 1))
        other_anchors = [(neg_anchor_name, neg_img_number_1, neg_img_number_2)
                         for neg_anchor_name, neg_img_number_1, neg_img_number_2 in positive_test_rows if neg_anchor_name != anchor_name]

        # Loading NEGATIVE target images for the task
        for i in range(negatives_per_task):
            neg_anchor_name, neg_img_number_1, _ = random.choice(other_anchors)
            negative_image = load_image(neg_anchor_name, neg_img_number_1)
            task.append((anchor_img, negative_image, 0))

        tasks.append(task)

    return tasks


def _execute_one_shot_tasks(model, tasks):
    """
    Executing all one-shot classification tasks.
    This function returns the total accuracy on all one-shot tasks.
    In addition it returns the task in which the model succeeded the most, and the task it was least accurate on.
    """

    success_cnt = 0

    best_result = 0
    task_with_best_result = -1
    worst_result = 1
    task_with_worst_result = -1

    for task in tasks:
        # Evaluating each task
        anchors = np.array([task_pair[0] for task_pair in task])
        targets = np.array([task_pair[1] for task_pair in task])

        prediction = model.predict([anchors, targets])
        current_res = np.argmax(prediction) == 0
        success_cnt += current_res

        # If current task had best/worst result - we keep it.
        if current_res == 1 and prediction[0] > best_result:
            best_result = prediction[0]
            task_with_best_result = (task, prediction)
        elif current_res == 0 and prediction[0] < worst_result:
            worst_result = prediction[0]
            task_with_worst_result = (task, prediction)

    accuracy = success_cnt / len(tasks)
    return accuracy, task_with_best_result, task_with_worst_result


def result_evaluation(model, test_X, test_Y, batch_size, one_shot_acc, worst_task, best_task):
    """
    Evaluating the model on the test-set
    """
    [loss, binary_acc, recall, percision, AUC] = model.evaluate(test_X, test_Y, batch_size=batch_size)
    print("***********************************************************************************")
    print(f'Evaluation parameters:\nbinary accuracy: {binary_acc}\nrecall: {recall}\npercision: {percision}\nAUC: {AUC}\nTest Loss: {loss}')
    print(f"One Shot Accuracy = {one_shot_acc}")
    print('One-Shot task with the best model prediction:')
    display_one_shot_task(best_task, 'best_task')
    print('One-Shot task with the worst model prediction:')
    display_one_shot_task(worst_task, 'worst_task')

    return loss, binary_acc, recall, percision, AUC



def train_and_evaluate_model(input_shape, batch_size, learning_rate, momentum, fc_regularizer, conv_regularizer, is_augmented, is_normalized, dropout=False, dense_size=4096):
    """
    Encapsulates the whole train-evaluation process.
    """
    optimizer = Adam(lr=learning_rate, beta_1=momentum)

    # Defining Callbacks
    checkpoint_filepath = './checkpoint/'
    os.makedirs(checkpoint_filepath, exist_ok=True)
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    log_dir = "./logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    learning_rate_scheduler = LearningRateScheduler(decay_schedule)
    early_stopper = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

    callbacks = [early_stopper, tensorboard_callback, learning_rate_scheduler, model_checkpoint_callback]

    # Parsing the data
    train_anchors, train_targets, train_positivities = make_norm_anchor_target_set(TRAIN_SET_PATH)
    test_anchors, test_targets, test_positivities = make_norm_anchor_target_set(TEST_SET_PATH)

    # Splitting training set to train/validation
    train_anchors, train_targets, train_positivities, \
    validation_anchors, validation_targets, validation_positivities = _shuffle_split_train_validation(train_anchors, train_targets, train_positivities, 0.2)
    validation_data = ([validation_anchors, validation_targets], validation_positivities)

    # Preprocessing:
    train_anchors, train_targets, train_positivities, validation_anchors, validation_targets, test_anchors, test_targets = \
        preprocessing(train_anchors, train_targets, train_positivities, validation_anchors, validation_targets, test_anchors, test_targets,
         Normalization=is_normalized, Augmatation=is_augmented, wise='featurewise')

    # Generating One-Shot tasks
    tasks = _make_one_shot_tasks(TEST_SET_PATH, 9)

    # Building the network
    model = build_siamese_model(input_shape, conv_regularizer=conv_regularizer, fc_regularizer=fc_regularizer, dropout=dropout, dense_size=dense_size)
    model.summary()

    print("************************************************")
    metrics = [BinaryAccuracy(), Recall(), Precision(), AUC()]
    model.compile(loss="binary_crossentropy", optimizer=optimizer,  metrics=metrics)
    loss = model.fit([train_anchors, train_targets], train_positivities, batch_size=batch_size, epochs=25,
                     validation_data=validation_data, callbacks=callbacks, shuffle=True)

    # Loading best model from checkpoint
    model = tf.keras.models.load_model(checkpoint_filepath)

    # Running model on One-Shot tasks
    test_one_shot_acc, one_shot_best_task, one_shot_worst_task = _execute_one_shot_tasks(model, tasks)

    # Evaluating model on test-set
    test_loss, test_binary_acc, test_recall, test_percision, test_AUC = result_evaluation(model, [test_anchors, test_targets], test_positivities, batch_size,
                      test_one_shot_acc, one_shot_worst_task, one_shot_best_task)

    csv_results = {"batch_size": batch_size, "learning rate": learning_rate, "momentum": momentum,
                   "fc_regularizer": fc_regularizer, "conv_regularizer": conv_regularizer, "dropout": dropout,
                   "test_accuracy": test_binary_acc, "test_recall": test_recall, "test_percision": test_percision,
                   "test_AUC": test_AUC, "test_one_shot_acc": test_one_shot_acc, "is_augmented": is_augmented, "is_normalized": is_normalized,
                   "log_dir": log_dir}

    with open("./res.csv", 'a+') as csv_file:
        fieldnames = list(csv_results.keys())
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=',')
        if len(csv_file.readlines()) == 0:
            writer.writeheader()
        writer.writerow(csv_results)


# Hyper Parameters
input_shape = (105, 105, 1)
batch_size = 128
learning_rate = 0.0001
momentum = 0.5
fc_regularizer = 0.0001
conv_regularizer = 0.001

is_augmented = False
is_normalized = False
train_and_evaluate_model(input_shape, batch_size, learning_rate, momentum, fc_regularizer, conv_regularizer, is_augmented, is_normalized, dropout=False, dense_size=2048)
