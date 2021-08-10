import math
import random
import time
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(14)
tf.random.set_seed(14)

class GAN_BB(BaseEstimator, TransformerMixin):

    def __init__(self, discriminator_network_params, generator_network_params, black_box_model, dataset_name, epochs=2000, batch_size=256, seed=42, dummy_fields_indices=None):

        self.discriminator_network_params = discriminator_network_params
        self.generator_network_params = generator_network_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dummy_fields_indices = dummy_fields_indices
        self.black_box_model = black_box_model
        self.dataset_name = dataset_name

        self.generator_model = None
        self.GAN_BB_model = None

        self.GAN_loss = []
        self.GAN_accuracy = []

        self.best_GAN_loss = math.inf
        self.best_weights_GAN = None

        self._build_GAN()

    def _create_generator_inputs(self, n):
        """
        Sampling a noise dataset as an input for the generator
        output is a matrix of [batch_size/2, gen_input_dim]
        """
        gen_input_dim = self.generator_network_params['input_dim']
        generator_inputs = np.random.normal(0, 1, size=[int(n), gen_input_dim])
        return generator_inputs

    def generate_n_samples(self, noise, C):
        """
        Sampling dataset from the generator output distribution
        output is a matrix of [batch_size/2, number of features in the real data]
        """
        fake_samples = self.generator_model.predict([noise, C])
        return fake_samples

    def _build_generator_model(self, input_noise, input_C):
        """
        Creating the generator part in the GAN model
        """
        generator_dimensions = self.generator_network_params['dimensions']
        generator_output_dim = self.generator_network_params['output_dim']
        generator_activations = self.generator_network_params['activations']
        generator_input_dim = self.generator_network_params['input_dim']

        merged = Concatenate(axis=1)([input_noise, input_C])
        dense = Dense(generator_dimensions[0], input_dim=generator_input_dim, activation=generator_activations[0], use_bias=True)(merged)

        for dimension, activation in zip(generator_dimensions[1:], generator_activations[1:]):
            dense = Dense(dimension, activation=activation)(dense)

        output = Dense(generator_output_dim, activation="sigmoid")(dense)
        self.generator_model = Model([input_noise, input_C], output)

        return output

    def _build_GAN(self):
        """
        Creating the GAN model
        """
        discriminator_dimensions = self.discriminator_network_params['dimensions']
        discriminator_activations = self.discriminator_network_params['activations']

        generator_input_dim = self.generator_network_params['input_dim']

        # NOTE: the confidence score input is taken twice here:
        # 1. First time is in input_C, which is used only by the generator.
        # 2. The second time is in input_mashed_C, which is the result of shuffling confidence-scores with Y values.
        #    This input is used by the discriminator, and it is shuffled with Y so it won't rely on the channel it gets
        #    the Y values from.
        input_noise = Input(shape=(generator_input_dim,))
        input_C = Input(shape=(1,))
        input_mashed_C = Input(shape=(1,))
        input_mashed_Y = Input(shape=(1,))

        generator_output = self._build_generator_model(input_noise, input_C)

        merged = Concatenate(axis=1)([generator_output, input_mashed_C, input_mashed_Y])

        dense = Dense(discriminator_dimensions[0], activation=discriminator_activations[0])(merged)
        for dimension, activation in zip(discriminator_dimensions[1:], discriminator_activations[1:]):
            dense = Dense(dimension, activation=activation)(dense)

        output = Dense(1, activation='sigmoid')(dense)
        self.GAN_BB_model = Model([input_noise, input_C, input_mashed_C, input_mashed_Y], output)
        self.GAN_BB_model.compile(loss="binary_crossentropy", optimizer=Adam(lr=1e-5), metrics=['accuracy'])

    def _mash(self, confidence_score, Y):
        """
        Shuffling confidence-score values with Y values in a random manner.
        """
        mashed_C = np.zeros((self.batch_size, 1))
        mashed_Y = np.zeros((self.batch_size, 1))
        target = np.zeros((self.batch_size, 1))

        indices_list = [i for i in range(self.batch_size)]
        random_half_indices = np.random.choice(indices_list, int(self.batch_size/2), replace=False)
        second_half_indices = [i for i in indices_list if i not in random_half_indices]

        mashed_C[random_half_indices] = Y[random_half_indices]
        mashed_Y[random_half_indices] = confidence_score[random_half_indices]

        mashed_C[second_half_indices] = confidence_score[second_half_indices]
        mashed_Y[second_half_indices] = Y[second_half_indices]
        target[second_half_indices] = 1
        return mashed_C, mashed_Y, target

    def _do_iteration(self, epoch):
        """
        Make one learning iteration of the whole GAN model.
        """

        # Generating noise and confidence-score
        confidence_score = np.random.uniform(0, 1, self.batch_size).reshape(self.batch_size,-1)
        noise = self._create_generator_inputs(self.batch_size)

        # Generating new samples.
        generated_samples = self.generate_n_samples(noise, confidence_score)
        generated_X = generated_samples[:,:-1]

        # Running the BlackBox model on the generated samples
        Y = self.black_box_model.predict_proba(generated_X)[:,0].reshape((self.batch_size, 1))

        # Shuffling C and Y before feeding them to the discriminator.
        mashed_C, mashed_Y, target = self._mash(confidence_score, Y)
        GAN_BB_loss, GAN_BB_accuracy = self.GAN_BB_model.train_on_batch([noise, confidence_score, mashed_C, mashed_Y], target)

        print(f"Finished epoch {epoch} with batch_size={self.batch_size}.\nGAN_BB loss = {GAN_BB_loss}\n")
        return GAN_BB_loss, GAN_BB_accuracy

    def fit(self, patiance, X=None, y=None):
        """
        Main fitting method of the model
        """

        if (self.GAN_BB_model == None):
            raise Exception("You should create the GAN model first (call create_model())")

        not_improved = 0
        for epoch in range(self.epochs):
            curr_GAN_loss, curr_GAN_accuracy = self._do_iteration(epoch)
            self.GAN_loss.append(curr_GAN_loss)
            self.GAN_accuracy.append(curr_GAN_accuracy)

            if curr_GAN_loss < self.best_GAN_loss:
                self.best_GAN_loss = curr_GAN_loss
                self.epoch_with_best_loss = epoch
                not_improved = 0
                self.best_weights_GAN = self.GAN_BB_model.get_weights()
            else:
                not_improved += 1

            # Early-stop
            if not_improved > patiance:
                break

        # Loading the weights of the best model
        self.GAN_BB_model.set_weights(self.best_weights_GAN)

        print(f"*** Epoch with best weights: {self.epoch_with_best_loss} ***")
        self.check_model_performance(1000)
        return self


    def plot_loss_graph(self, loss, model_type, dataset):
        """
        Plotting loss graph
        """
        fig = plt.figure()
        title = f"Loss function of {model_type}"

        plt.plot(range(len(loss)), loss)
        plt.title = title
        plt.xlabel = "Epoch"
        plt.ylabel = "Loss"
        plt.savefig(fname=f"figures/{dataset}_{model_type}_loss")
        plt.close()

    def plot_acc_graph(self, acc, model_type, dataset):
        """
        Plotting accuracy graph
        """
        fig = plt.figure()
        title = f"Accuracy function of {model_type}"

        plt.plot(range(len(acc)), acc)
        plt.title = title
        plt.xlabel = "Epoch"
        plt.ylabel = "Accuracy"
        plt.savefig(fname=f"figures/{dataset}_{model_type}_acc")
        plt.close()

    def check_model_performance(self, n_samples):
        """
        Evaluates the model's performance by testing the discriminator of some generated samples.
        """
        # Generating N samples using our trained GAN
        confidence_score = np.random.uniform(0, 1, n_samples).reshape(n_samples,-1)
        noise = self._create_generator_inputs(n_samples)

        generated_samples = self.generate_n_samples(noise, confidence_score)
        generated_X = generated_samples[:,:-1]

        # Trying to predict the class with our BlackBox model, on the generated samples
        Y = self.black_box_model.predict_proba(generated_X)[:,0].reshape((n_samples, 1))

        # Feed the discriminator these generated samples along with the confidence scores and the BlackBox predictions.
        Y_kova = self.GAN_BB_model.predict([generated_samples, confidence_score, confidence_score, Y])

        # Calculating the number of time the discriminator was fooled by the generator
        total_fools = np.where(Y_kova < 0.5)
        print(f"The Discriminator was fooled {len(total_fools)/n_samples}% of the time")
