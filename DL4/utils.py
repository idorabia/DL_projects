from scipy.io import arff
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from DL4.GAN_with_BB_model import GAN_BB


def load_data (file_name):
    """
    Loading arff file
    """
    arff_data = pd.DataFrame(arff.loadarff(file_name)[0])
    return arff_data


def preprocessing_data(data, dataset_name):
    """
    Scaling and one hot encoding features. Label encoding target label.
    """
    if dataset_name == "diabetes":
        # Scailing the numeric values:
        numeric_features = pd.DataFrame(data, columns=data.columns.array[:8]).to_numpy()
        scaled_numeric_features = pd.DataFrame(MinMaxScaler().fit_transform(numeric_features), columns=data.columns.array[:8]).to_numpy()

        # Label encoding for the class labels:
        categorical_data = pd.DataFrame(data, columns=["class"])
        enc = preprocessing.LabelEncoder()
        target_feature = enc.fit_transform(categorical_data)
        target_feature = target_feature.reshape((len(target_feature), 1))
        return scaled_numeric_features, target_feature

    if dataset_name == "german_credit":
        categorical_columns = ["1", "3", "4", "6", "7", "8", "9", "10", "11", "12", "14", "15", "16", "17", "18", "19", "20"]
        numeric_columns = [col for col in data.columns[:20] if col not in categorical_columns]

        # Scailing the numeric values:
        numeric_features = pd.DataFrame(data[numeric_columns]).to_numpy()
        scaled_numeric_features = pd.DataFrame(MinMaxScaler().fit_transform(numeric_features), columns=numeric_columns).to_numpy()

        # One hot encoding for the class labels:
        enc = preprocessing.OneHotEncoder()
        dummy_featured_data = enc.fit_transform(data[categorical_columns]).toarray()

        # Good or bad customer:
        customer_quality = pd.DataFrame(data["21"])
        enc = preprocessing.LabelEncoder()
        target_feature = enc.fit_transform(customer_quality)
        target_feature = target_feature.reshape((len(target_feature), 1))

        preprocessed_features = np.concatenate((scaled_numeric_features, dummy_featured_data), axis=1)
        return preprocessed_features, target_feature


def get_random_forest_model(X_train, X_test, y_train, seed, dataset):
    """
    Training a RandomForest model on one of the datasets.
    Generating a distribution plot of the prediction probabilities on the test-set.
    """
    rf = RandomForestClassifier(n_estimators=25, max_depth=3, random_state=seed).fit(X_train, y_train)

    test_proba = rf.predict_proba(X_test)[:,0]
    min_proba = np.min(test_proba)
    max_proba = np.max(test_proba)
    mean_proba = np.mean(test_proba)
    print(f"RandomForest Performance on {dataset}: min = {min_proba}\tmax = {max_proba}\tmean = {mean_proba}")

    fig = plt.figure()
    title = f"Distribution of RandomForest confidence-level"
    sns.distplot(test_proba, hist=False)
    plt.legend()
    plt.title = title
    plt.savefig(fname=f"figures/{dataset}_RF_acc")
    plt.close()

    return rf


def train_and_evaluate_BB_model(dataset_name, bb_model, preprocessed_data, target_data, seed, discriminator_BB_network_params, generator_BB_network_params):
    """
    Train & Evaluate the BB-GAN model.
    """
    X_train, X_test, y_train, y_test = train_test_split(preprocessed_data,
                                                        target_data,
                                                        test_size=0.3,
                                                        random_state=seed)

    # Training the BB_GAN and saving best model
    gan_BB = GAN_BB(discriminator_BB_network_params, generator_BB_network_params, bb_model,
                             dataset_name=dataset_name, epochs=5000)
    gan_BB = gan_BB.fit(patiance=1000)

    gan_BB.plot_loss_graph(gan_BB.GAN_loss, "GAN-BB", dataset_name)
    gan_BB.plot_acc_graph(gan_BB.GAN_accuracy, "GAN-BB", dataset_name)
    tf.keras.models.save_model(gan_BB.GAN_BB_model, f'checkpoint/{dataset_name}/GAN_BB/')

    # Generating 1000 samples using our GAN
    confidence_score = np.random.uniform(0, 1, 1000).reshape(1000, -1)
    noise = gan_BB._create_generator_inputs(1000)
    generated_samples = gan_BB.generate_n_samples(noise, confidence_score)
    generated_X = generated_samples[:, :-1]
    generated_y = generated_samples[:, -1]
    label_1_indices = np.where(generated_y > 0.5)[0]
    label_0_indices = np.where(generated_y < 0.5)[0]

    # Trying to predict the class with our BlackBox model, on the generated samples
    Y = bb_model.predict_proba(generated_X)[:,0].reshape((1000, 1))
    Y_label_1 = Y[label_1_indices]
    Y_label_0 = Y[label_0_indices]

    # Trying to predict the class with our BlackBox model, on the REAL test samples
    test_proba = bb_model.predict_proba(X_test)[:, 0]
    test_proba_label_1 = test_proba[np.where(y_test == 1)[0]]
    test_proba_label_0 = test_proba[np.where(y_test == 0)[0]]

    # Comparing the distributions of the BlackBox predictions on the test samples and the generated samples.
    fig = plt.figure()
    title = f"Distribution of BlackBox predictions on original samples (test) and generated samples"
    sns.distplot(Y, label="BlackBox predictions on generated samples", hist=False)
    sns.distplot(test_proba, label="BlackBox predictions on real test samples", hist=False)
    plt.title = title
    plt.legend()
    plt.savefig(fname=f"figures/{dataset_name}_all_dist")
    plt.close()

    fig = plt.figure()
    title = f"Distribution of BlackBox predictions on original samples (test) and generated samples (label=1)"
    sns.distplot(Y_label_1, label="BlackBox predictions on generated samples", hist=False)
    sns.distplot(test_proba_label_1, label="BlackBox predictions on real test samples", hist=False)
    plt.title = title
    plt.legend()
    plt.savefig(fname=f"figures/{dataset_name}_label_1_dist")
    plt.close()

    fig = plt.figure()
    title = f"Distribution of BlackBox predictions on original samples (test) and generated samples (label=0)"
    sns.distplot(Y_label_0, label="BlackBox predictions on generated samples", hist=False)
    sns.distplot(test_proba_label_0, label="BlackBox predictions on real test samples", hist=False)
    plt.title = title
    plt.legend()
    plt.savefig(fname=f"figures/{dataset_name}_label_0_dist")
    plt.close()
    return gan_BB