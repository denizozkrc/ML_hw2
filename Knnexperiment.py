import pickle
from Distance import Distance
from Part1.Knn import KNN
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

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


for i, (train_indices, test_indices) in enumerate(kfold.split(dataset, labels)):
    current_train = dataset[train_indices]
    current_train_labels = labels[train_indices]
    current_test = dataset[test_indices]
    current_test_labels = labels[test_indices]
    predicted_label = np.zeros_like(current_test_labels)

    for j, conf in enumerate(hyperparameter_configurations):
        knn = KNN(dataset=current_train, data_label=current_train_labels, similarity_function=conf[0], similarity_function_parameters=conf[1], K=conf[2])

        for k in range(len(current_test)):
            predicted_label[k] = knn.predict(current_test[k])
        accuracy[j] += accuracy_score(current_test_labels, predicted_label)
    # acc = accuracy[0]/(i+1)
    # print("Accuracy %.2f" % acc)

for j in range(hyperparameter_configurations_len):
    accuracy[j] = accuracy[j]/(n_splits*n_repeats)
    print("Accuracy %.2f" % accuracy[j])
