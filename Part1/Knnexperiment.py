import pickle
from Distance import Distance
from Knn import KNN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

# the data is already preprocessed
dataset, labels = pickle.load(open("datasets/part1_dataset.data", "rb"))

n_splits = 10
n_repeats = 5

kfold = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=None)

param_grid_cosine = {'fnc': [Distance.calculateCosineDistance], 'no_param': [None], 'K': [5, 7, 10]}
hyperparameter_configurations_cosine = [(c, k, l) for c in param_grid_cosine['fnc'] for k in param_grid_cosine['no_param'] for l in param_grid_cosine['K']]

param_grid_mink = {'fnc': [Distance.calculateMinkowskiDistance], 'p_val': [1, 2, 3], 'K': [5, 7, 10]}
hyperparameter_configurations_mink = [(c, k, l) for c in param_grid_mink['fnc'] for k in param_grid_mink['p_val'] for l in param_grid_mink['K']]

param_grid_mah = {'fnc': [Distance.calculateMahalanobisDistance], 'S_minus_1': [None], 'K': [5, 7, 10]}
hyperparameter_configurations_mah = [(c, k, l) for c in param_grid_mah['fnc'] for k in param_grid_mah['S_minus_1'] for l in param_grid_mah['K']]

hyperparameter_configurations_cosine_len = len(hyperparameter_configurations_cosine)
hyperparameter_configurations_mink_len = len(hyperparameter_configurations_mink)
hyperparameter_configurations_mah_len = len(hyperparameter_configurations_mah)


accuracy_cos = [[] for _ in range(hyperparameter_configurations_cosine_len)]
accuracy_mink = [[] for _ in range(hyperparameter_configurations_mink_len)]
accuracy_mah = [[] for _ in range(hyperparameter_configurations_mah_len)]

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
        accuracy_cos[j].append(accuracy_score(current_test_labels, predicted_label))

    for j, conf in enumerate(hyperparameter_configurations_mink):
        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=conf[0], similarity_function_parameters=conf[1], K=conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy_mink[j].append(accuracy_score(current_test_labels, predicted_label))

    for j, conf in enumerate(hyperparameter_configurations_mah):
        S_minus_1 = np.linalg.inv(np.cov(np.transpose(current_test)))
        new_conf = (conf[0], S_minus_1, conf[2])

        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=new_conf[0], similarity_function_parameters=new_conf[1], K=new_conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy_mah[j].append(accuracy_score(current_test_labels, predicted_label))


def meanAndCi(accuracies):
    mean_accuracy = np.mean(np.array(accuracies))
    std_dev = np.std(accuracies)
    ci = 1.96 * std_dev / np.sqrt(n_repeats*n_splits)
    return mean_accuracy, ci


for j in range(hyperparameter_configurations_cosine_len):
    mean_accuracy, ci = meanAndCi(accuracy_cos[j])
    print(f"Cosine Config {hyperparameter_configurations_cosine[j][1:3]}: Accuracy = {mean_accuracy:.4f}, 95% CI = {ci:.4f}")

for j in range(hyperparameter_configurations_mink_len):
    mean_acc, ci = meanAndCi(accuracy_mink[j])
    print(f"Minkowski Config {hyperparameter_configurations_mink[j][1:3]}: Accuracy = {mean_acc:.4f}, 95% CI = {ci:.4f}")

for j in range(hyperparameter_configurations_mah_len):
    mean_acc, ci = meanAndCi(accuracy_mah[j])
    print(f"Mahalanobis Config {hyperparameter_configurations_mah[j][1:3]}: Accuracy = {mean_acc:.4f}, 95% CI = {ci:.4f}")
