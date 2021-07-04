import math
import numpy as np
import tensorflow as tf
from sklearn.base import BaseEstimator, TransformerMixin
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

np.random.seed(14)
tf.random.set_seed(14)

class GAN(BaseEstimator, TransformerMixin):

    def __init__(self, discriminator_network_params, generator_network_params, epochs=2000, batch_size=256, seed=42, dummy_fields_indices=None):

        self.discriminator_network_params = discriminator_network_params
        self.generator_network_params = generator_network_params
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed
        self.dummy_fields_indices = dummy_fields_indices

        self.generator_model = None
        self.discriminator_model = None
        self.GAN_model = None
        self.data = None

        self.GAN_loss = []
        self.gen_loss = []
        self.gen_acc = []
        self.disc_loss = []
        self.disc_acc = []

        self.best_GAN_loss = math.inf
        self.best_weights_GAN = None
        self.best_weights_discriminator = None
        self.best_weights_generator = None

        self.create_model()

    def _create_generator_inputs(self, n):
        """
        Sampling a noise dataset as an input for the generator
        output is a matrix of [batch_size/2, gen_input_dim]
        """
        gen_input_dim = self.generator_network_params['input_dim']
        generator_inputs = np.random.normal(0, 1, size=[int(n), gen_input_dim])
        return generator_inputs

    def _generate_n_samples(self, n):
        """
        Sampling dataset from the generator output distribution
        output is a matrix of [batch_size/2, number of features in the real data]
        """
        generator_input = self._create_generator_inputs(n)
        fake_samples = self.generator_model.predict(generator_input)

        # Rounding one-hot fields to 0 or 1
        if self.dummy_fields_indices != None:
            for index in self.dummy_fields_indices:
                fake_samples[:,index] = np.where(fake_samples[:,index] > 0.5, 1, 0)

        return fake_samples, np.zeros((int(self.batch_size/2), 1))

    def _real_samples(self):
        """
        Sampling dataset from the real data
        output is a matrix of [batch_size, number of features in the real data]
        """
        real_samples_indices = np.random.choice(self.data.shape[0], size=int(self.batch_size / 2), replace=False)
        real_samples = self.data[real_samples_indices]

        return real_samples, np.ones((int(self.batch_size/2), 1))

    def _build_generator_model(self):
        """
        Creating a generator model
        """
        dimensions = self.generator_network_params['dimensions']
        input_dim = self.generator_network_params['input_dim']
        output_dim = self.generator_network_params['output_dim']
        activations = self.generator_network_params['activations']

        model = Sequential()

        model.add(Dense(dimensions[0], activations[0], input_dim=input_dim))

        for dimension, activation in zip(dimensions[1:], activations[1:]):
            model.add(Dense(dimension, activation))

        model.add(Dense(output_dim, activation='sigmoid'))

        print("Generator Model:\n")
        model.summary()
        self.generator_model = model

    def _build_discriminator_model(self):
        """
        Creating a discriminator model
        """
        dimensions = self.discriminator_network_params['dimensions']
        input_dim = self.discriminator_network_params['input_dim']
        activations = self.discriminator_network_params['activations']

        model = Sequential()

        model.add(Dense(dimensions[0], activations[0], input_dim=input_dim))

        for dimension, activation in zip(dimensions[1:], activations[1:]):
            model.add(Dense(dimension, activation))

        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-5), metrics=['accuracy'])

        print("Discriminator Model:\n")
        model.summary()
        self.discriminator_model = model

    def _build_GAN(self):
        """
        Creating GAN model by sequencing the generator and the discriminator.
        """
        self.discriminator_model.trainable = False
        GAN_model = Sequential()
        GAN_model.add(self.generator_model)
        GAN_model.add(self.discriminator_model)
        GAN_model.compile(optimizer=Adam(lr=1e-5), loss="binary_crossentropy", metrics=['accuracy'])
        self.GAN_model = GAN_model

    def create_model(self):
        self._build_generator_model()
        self._build_discriminator_model()
        self._build_GAN()


    def _do_iteration(self, epoch):
        """
        Make one learning iteration of the whole GAN model.
        """

        # Getting half-batch of real samples and generated samples
        real_X, real_y = self._real_samples()
        generated_X, generated_y = self._generate_n_samples(self.batch_size/2)

        # Training the discriminator first
        loss_for_real, accuracy_for_real = self.discriminator_model.train_on_batch(real_X, real_y)
        loss_for_generated, accuracy_for_generated = self.discriminator_model.train_on_batch(generated_X, generated_y)
        curr_avg_disc_loss = (loss_for_real + loss_for_generated)/2
        curr_avg_disc_accuracy = (accuracy_for_real + accuracy_for_generated)/2
        self.disc_loss.append(curr_avg_disc_loss)
        self.disc_acc.append(curr_avg_disc_accuracy)

        # Generating noise for the generator
        GAN_X_1 = self._create_generator_inputs(self.batch_size/2)
        GAN_X_2 = self._create_generator_inputs(self.batch_size/2)
        GAN_X = np.concatenate((GAN_X_1, GAN_X_2))
        GAN_y = np.ones((self.batch_size, 1))

        # Training the generator
        curr_generator_loss, curr_generator_accuracy = self.GAN_model.train_on_batch(GAN_X, GAN_y)
        self.gen_loss.append(curr_generator_loss)
        self.gen_acc.append(curr_generator_accuracy)

        # Boosting Step:
        #   If one of the two (G and D) is weaker than the other -> it keeps training for another 3 iterations.
        #   The idea is to close the gap between the discriminator and the generator as much as possible.
        for i in range(3):
            if self.gen_loss[-1] > self.disc_loss[-1]:
                GAN_X_1 = self._create_generator_inputs(self.batch_size/2)
                GAN_X_2 = self._create_generator_inputs(self.batch_size/2)
                GAN_X = np.concatenate((GAN_X_1, GAN_X_2))
                GAN_y = np.ones((self.batch_size, 1))
                self.GAN_model.train_on_batch(GAN_X, GAN_y)
            else:
                real_X, real_y = self._real_samples()
                generated_X, generated_y = self._generate_n_samples(self.batch_size/2)
                self.discriminator_model.train_on_batch(real_X, real_y)
                self.discriminator_model.train_on_batch(generated_X, generated_y)

        print(f"Finished epoch {epoch} with batch_size={self.batch_size}.\n"
              f"Generator-loss = {curr_generator_loss}\t"
              f"Generator-accuracy = {curr_generator_accuracy}\n"
              f"Discriminator-loss = {curr_avg_disc_loss}\t"
              f"Discriminator-accuracy = {curr_avg_disc_accuracy}\n")
        return curr_avg_disc_loss, curr_generator_loss

    def fit(self, X, patiance, y=None):
        """
        Main fitting method of the model
        """

        if (self.GAN_model == None):
            raise Exception("You should create the GAN model first (call create_model())")
        self.data = X
        not_improved = 0

        for epoch in range(self.epochs):
            discriminator_loss, generator_loss = self._do_iteration(epoch)
            last_gen_loss = generator_loss
            last_disc_loss = discriminator_loss

            # Calculating the loss of the GAN as the sum of losses plus the gap between G and D.
            gap_factor = abs(last_gen_loss - last_disc_loss)
            curr_GAN_loss = last_gen_loss + last_disc_loss + gap_factor

            if curr_GAN_loss < self.best_GAN_loss:
                self.best_GAN_loss = curr_GAN_loss
                self.epoch_with_best_loss = epoch
                not_improved = 0
                self.best_weights_GAN = self.GAN_model.get_weights()
                self.best_weights_discriminator = self.discriminator_model.get_weights()
                self.best_weights_generator = self.generator_model.get_weights()
            else:
                not_improved += 1

            # Early-stop
            if not_improved > patiance:
                break

        # Loading the weights of the best model
        self.GAN_model.set_weights(self.best_weights_GAN)
        self.discriminator_model.set_weights(self.best_weights_discriminator)
        self.generator_model.set_weights(self.best_weights_generator)

        self.check_model_performance(100)
        print(f"*** Epoch with best weights: {self.epoch_with_best_loss} ***")
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

        # Generates N samples and run the discriminator on them.
        generated_samples = self._generate_n_samples(n_samples)[0]
        discriminator_pred = self.discriminator_model.predict(generated_samples)

        # Calculating the euclidean distance of each real sample from the rest of the real samples.
        real_euclidean_distances = [np.linalg.norm(self.data[i] - np.delete(self.data, [i], axis=0).mean(axis=0))
                                          for i in range(len(self.data))]
        real_euclidean_dist_std = np.std(real_euclidean_distances)
        real_euclidean_dist_mean = np.mean(real_euclidean_distances)

        # Constants for the loop
        global_minimal_euclidean_dist = np.inf
        best_fooled_sample = None
        global_maximal_euclidean_dist = -1
        worst_fooled_sample = None

        tuples = []
        for i in range(len(generated_samples)):
            minimal_euclidean_dist = np.inf
            closest_real_sample = None
            ind_of_closest_real_sample = None

            # For each generated samples, check if fooled the discriminator.
            if discriminator_pred[i] > 0.5:
                disc_fooled = 1
            else:
                disc_fooled = 0

            # In addition, find its closest real sample
            for j in range(len(self.data)):
                current_euclidean_dist = np.linalg.norm(generated_samples[i] - self.data[j])
                if current_euclidean_dist < minimal_euclidean_dist:
                    minimal_euclidean_dist = current_euclidean_dist
                    closest_real_sample = self.data[j]
                    ind_of_closest_real_sample = j

            # Calculates its distance from the rest (euclidean distance)
            generated_dist_from_all_real = np.linalg.norm(generated_samples[i] - self.data.mean(axis=0))
            all_data_wo_closest = np.delete(self.data, [ind_of_closest_real_sample], axis=0)
            all_data_wo_closest_mean = all_data_wo_closest.mean(axis=0)
            real_euclidean_dist_from_mean = np.linalg.norm(closest_real_sample - all_data_wo_closest_mean)

            # Data structure to store all this data for each generated sample
            tup = (generated_samples[i], closest_real_sample, generated_dist_from_all_real, minimal_euclidean_dist, real_euclidean_dist_from_mean, disc_fooled)
            tuples.append(tup)

            if disc_fooled and minimal_euclidean_dist < global_minimal_euclidean_dist:
                global_minimal_euclidean_dist = minimal_euclidean_dist
                best_fooled_sample = tup
            elif not disc_fooled and minimal_euclidean_dist > global_maximal_euclidean_dist:
                global_maximal_euclidean_dist = minimal_euclidean_dist
                worst_fooled_sample = tup

        total_fooled = sum([tup[5] for tup in tuples])
        print("***************************************************************************************************")
        print("***************************************************************************************************")
        print(f"Percentage of generated samples that fooled the Discriminator: {(total_fooled / n_samples) * 100}%\n")
        print(f"Mean euclidean distance of real samples: {real_euclidean_dist_mean}\n")
        print(f"Standard-Deviation of euclidean distances of real samples: {real_euclidean_dist_std}\n")

        print(f"Example of a normalized sample that fooled the Discriminator: {best_fooled_sample[0]}\nClosest real-sample: {best_fooled_sample[1]}\n"
              f"Euclidean distance between generated and all real samples: {best_fooled_sample[2]}\n"
              f"Euclidean distance between generated and real: {best_fooled_sample[3]}\n"
              f"Euclidean distance between real and other real samples: {best_fooled_sample[4]}")

        if worst_fooled_sample != None:
            print(f"\nExample of a normalized sample that DID NOT fooled the Discriminator: {worst_fooled_sample[0]}\nClosest real-sample: {worst_fooled_sample[1]}\n"
                  f"Euclidean distance between generated and all real samples: {worst_fooled_sample[2]}\n"
                  f"Euclidean distance between generated and real: {worst_fooled_sample[3]}\n"
                  f"Euclidean distance between real and other real samples: {worst_fooled_sample[4]}")
        print("***************************************************************************************************")
        print("***************************************************************************************************")

        return tuples
