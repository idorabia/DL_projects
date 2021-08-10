from sklearn.model_selection import train_test_split
import utils
from DL4.GAN_model import GAN
import tensorflow as tf
SEED = 42



# Data handling:
# German credit dataset:
german_credit_data = utils.load_data("german_credit.arff")
german_credit_preprocessed_data, german_credit_target = utils.preprocessing_data(german_credit_data, "german_credit")

# Diabetes dataset:
diabetes_data = utils.load_data("diabetes.arff")
diabetes_preprocessed_data, diabetes_target = utils.preprocessing_data(diabetes_data, "diabetes")

#
# ####################################################################################
# ###################################### PART 1 ######################################
# ####################################################################################
#
# # Training Diabetes GAN model:
# discriminator_network_params = {
#     'dimensions': [50, 30, 15, 10, 5],
#     'input_dim': diabetes_preprocessed_data.shape[1],
#     'activations': ['relu', 'relu', 'relu', 'relu']
# }
# generator_network_params = {
#     'dimensions': [50, 70, 50],
#     'input_dim': diabetes_preprocessed_data.shape[1],
#     'output_dim': diabetes_preprocessed_data.shape[1],
#     'activations': ['relu', 'relu', 'relu']
# }
# diabetes_gan = GAN(discriminator_network_params, generator_network_params, epochs=5000)
# diabetes_gan.data = diabetes_preprocessed_data
# diabetes_gan = diabetes_gan.fit(X=diabetes_gan.data, patiance=1000)
# diabetes_gan.plot_acc_graph(diabetes_gan.gen_acc, "Generator", "diabetes")
# diabetes_gan.plot_acc_graph(diabetes_gan.disc_acc, "Discriminator", "diabetes")
# diabetes_gan.plot_loss_graph(diabetes_gan.gen_loss, "Generator", "diabetes")
# diabetes_gan.plot_loss_graph(diabetes_gan.disc_loss, "Discriminator", "diabetes")
#
# tf.keras.models.save_model(diabetes_gan.GAN_model, 'checkpoint/diabetes/GAN/')
# tf.keras.models.save_model(diabetes_gan.generator_model, 'checkpoint/diabetes/Gen/')
# tf.keras.models.save_model(diabetes_gan.discriminator_model, 'checkpoint/diabetes/Disc/')














# Training German-Credit GAN model:
discriminator_network_params = {
    'dimensions': [80, 40, 20, 10, 5],
    'input_dim': german_credit_preprocessed_data.shape[1],
    'activations': ['relu', 'relu', 'relu', 'relu']
}
generator_network_params = {
    'dimensions': [80, 120, 80],
    'input_dim': german_credit_preprocessed_data.shape[1],
    'output_dim': german_credit_preprocessed_data.shape[1],
    'activations': ['relu', 'relu', 'relu']
}

german_gan = GAN(discriminator_network_params, generator_network_params, epochs=5000, dummy_fields_indices=[i for i in range(3, 71)])
german_gan.data = german_credit_preprocessed_data
german_gan.fit(X=german_gan.data, patiance=1000)
german_gan.plot_acc_graph(german_gan.gen_acc, "Generator", "german_credit")
german_gan.plot_acc_graph(german_gan.disc_acc, "Discriminator", "german_credit")
german_gan.plot_loss_graph(german_gan.gen_loss, "Generator", "german_credit")
german_gan.plot_loss_graph(german_gan.disc_loss, "Discriminator", "german_credit")

tf.keras.models.save_model(german_gan.GAN_model, 'checkpoint/german/GAN/')
tf.keras.models.save_model(german_gan.generator_model, 'checkpoint/german/Gen/')
tf.keras.models.save_model(german_gan.discriminator_model, 'checkpoint/german/Disc/')















####################################################################################
###################################### PART 2 ######################################
####################################################################################


discriminator_BB_network_params = {
    'dimensions': [50, 30, 15, 10, 5],
    'input_dim': diabetes_preprocessed_data.shape[1] + 2,
    'activations': ['relu', 'relu', 'relu', 'relu']
}
generator_BB_network_params = {
    'dimensions': [50, 70, 50],
    'input_dim': diabetes_preprocessed_data.shape[1]+1,
    'output_dim': diabetes_preprocessed_data.shape[1]+1,
    'activations': ['relu', 'relu', 'relu']
}

diabetes_X_train, diabetes_X_test, diabetes_y_train, diabetes_y_test = train_test_split(diabetes_preprocessed_data, diabetes_target, test_size=0.3, random_state=SEED)
diabetes_rf = utils.get_random_forest_model(diabetes_X_train, diabetes_X_test, diabetes_y_train, SEED, 'diabetes')

diabetes_gan_BB = utils.train_and_evaluate_BB_model('diabetes', diabetes_rf, diabetes_preprocessed_data, diabetes_target,
                                                    SEED, discriminator_BB_network_params, generator_BB_network_params)





discriminator_BB_network_params = {
    'dimensions': [80, 40, 20, 10, 5],
    'input_dim': german_credit_preprocessed_data.shape[1]+2,
    'activations': ['relu', 'relu', 'relu', 'relu']
}
generator_BB_network_params = {
    'dimensions': [80, 120, 80],
    'input_dim': german_credit_preprocessed_data.shape[1] + 1,
    'output_dim': german_credit_preprocessed_data.shape[1] + 1,
    'activations': ['relu', 'relu', 'relu']
}

german_credit_X_train, german_credit_X_test, german_credit_y_train, german_credit_y_test = \
    train_test_split(german_credit_preprocessed_data, german_credit_target, test_size=0.3, random_state=SEED)
german_credit_rf = utils.get_random_forest_model(german_credit_X_train, german_credit_X_test, german_credit_y_train,
                                                 SEED, 'german-credit')

german_credit_gan_BB = utils.train_and_evaluate_BB_model('german-credit', german_credit_rf,
                                                         german_credit_preprocessed_data, german_credit_target,
                                                         SEED, discriminator_BB_network_params, generator_BB_network_params)

