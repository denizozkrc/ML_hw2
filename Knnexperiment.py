import pickle
from Distance import Distance
from Part1.Knn import KNN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np
import scipy.stats as st

# the data is already preprocessed
dataset, labels = pickle.load(open("datasets/part1_dataset.data", "rb"))


n_splits = 10
n_repeats = 5
hyperparameter_configurations = [[Distance.calculateCosineDistance, None, 15],
                                 [Distance.calculateCosineDistance, None, 10],
                                 [Distance.calculateMinkowskiDistance, None, 15],
                                 [Distance.calculateMinkowskiDistance, None, 10],
                                 [Distance.calculateCosineDistance, None, 5],
                                 [Distance.calculateCosineDistance, None, 5]]
hyperparameter_configurations_len = len(hyperparameter_configurations)


kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

accuracy = np.zeros(hyperparameter_configurations_len)

param_grid_cosine = {'fnc': [Distance.calculateCosineDistance], 'no_param': [None], 'K': [5,7,10]}
hyperparameter_configurations_cosine = [(c, k, l) for c in param_grid_cosine['fnc'] for k in param_grid_cosine['no_param'] for l in param_grid_cosine['K']]

param_grid_mink = {'fnc': [Distance.calculateMinkowskiDistance], 'p_val': [1, 2, 3], 'K': [5,7,10]}
hyperparameter_configurations_mink = [(c, k, l) for c in param_grid_mink['fnc'] for k in param_grid_mink['p_val'] for l in param_grid_mink['K']]

param_grid_mah = {'fnc': [Distance.calculateMahalanobisDistance], 'S_minus_1': [None], 'K': [5,7,10]}
hyperparameter_configurations_mah = [(c, k, l) for c in param_grid_mah['fnc'] for k in param_grid_mah['S_minus_1'] for l in param_grid_mah['K']]

hyperparameter_configurations_cosine_len = len(hyperparameter_configurations_cosine)
hyperparameter_configurations_mink_len = len(hyperparameter_configurations_mink)
hyperparameter_configurations_mah_len = len(hyperparameter_configurations_mah)

accuracy_cos = np.zeros(hyperparameter_configurations_cosine_len)
accuracy_mink = np.zeros(hyperparameter_configurations_mink_len)
accuracy_mah = np.zeros(hyperparameter_configurations_mah_len)

for i, (train_indices, test_indices) in enumerate(kfold.split(dataset, labels)):
    current_train = dataset[train_indices]
    current_train_labels = labels[train_indices]
    current_test = dataset[test_indices]
    current_test_labels = labels[test_indices]
    predicted_label = np.zeros_like(current_test_labels)
    for j, conf in enumerate(hyperparameter_configurations_cosine):
        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=conf[0], similarity_function_parameters=conf[1], K=conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy_cos[j] += accuracy_score(current_test_labels, predicted_label)

    for j, conf in enumerate(hyperparameter_configurations_mink):
        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=conf[0], similarity_function_parameters=conf[1], K=conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy_mink[j] += accuracy_score(current_test_labels, predicted_label)

    for j, conf in enumerate(hyperparameter_configurations_mah):
        S_minus_1 = np.linalg.inv(np.cov(np.transpose(current_test)))
        new_conf = (conf[0], S_minus_1, conf[2])

        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=new_conf[0], similarity_function_parameters=new_conf[1], K=new_conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy_mah[j] += accuracy_score(current_test_labels, predicted_label)

for j in range(hyperparameter_configurations_cosine_len):
    accuracy_cos[j] = accuracy_cos[j]/(n_splits*n_repeats)
    print("Accuracy cos %i %.2f" % (j, accuracy_cos[j]))

for j in range(hyperparameter_configurations_mink_len):
    accuracy_mink[j] = accuracy_mink[j]/(n_splits*n_repeats)
    print("Accuracy min %i %.2f" % (j, accuracy_mink[j]))

for j in range(hyperparameter_configurations_mah_len):
    accuracy_mah[j] = accuracy_mah[j]/(n_splits*n_repeats)
    print("Accuracy mah %i %.2f" % (j, accuracy_mah[j]))
