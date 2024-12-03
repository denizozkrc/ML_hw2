import numpy as np


class KNN:
    def __init__(self, dataset, data_label, similarity_function, similarity_function_parameters=None, K=1):
        """
        :param dataset: dataset on which KNN is executed, 2D numpy array
        :param data_label: class labels for each data sample, 1D numpy array
        :param similarity_function: similarity/distance function, Python function
        :param similarity_function_parameters: auxiliary parameter or parameter array for distance metrics
        :param K: how many neighbors to consider, integer
        """
        self.K = K
        self.dataset = dataset
        self.dataset_label = data_label
        self.similarity_function = similarity_function
        self.similarity_function_parameters = similarity_function_parameters

    def predict(self, instance):
        closest_data_point_indexes = np.zeros(self.K)
        closest_data_points_distances = np.full(self.K, np.inf)
        closest_data_points_labels = np.zeros(self.K)
        index_of_max_distance = 0  # index in closest_data_points_distances
        max_number_of_label = 0

        number_of_training_samples = np.shape(self.dataset)[0]
        for sample_num in range(number_of_training_samples):
            dist = self.similarity_function(instance, self.dataset[sample_num], self.similarity_function_parameters)  # if (self.similarity_function_parameters) else self.similarity_function(instance, self.dataset[sample_num])
            if dist < closest_data_points_distances[index_of_max_distance]:
                closest_data_point_indexes[index_of_max_distance] = sample_num
                closest_data_points_distances[index_of_max_distance] = dist
                closest_data_points_labels[index_of_max_distance] = self.dataset_label[sample_num]
                index_of_max_distance = np.argmax(closest_data_points_distances)

        # arr, indices = np.unique(, return_index=True)
        for i in range(self.K):
            compare_array = np.full((1, self.K), closest_data_points_labels[i])
            occurence = np.count_nonzero(compare_array == closest_data_points_labels)
            if occurence > max_number_of_label:
                label = closest_data_points_labels[i]
                max_number_of_label = occurence

        return label
