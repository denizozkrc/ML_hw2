import numpy as np


class PCA:
    def __init__(self, projection_dim: int):
        """
        Initializes the PCA method
        :param projection_dim: the projection space dimensionality
        """
        self.projection_dim = projection_dim
        # keeps the projection matrix information
        self.projection_matrix = None

    def fit(self, x: np.ndarray) -> None:
        """
        Applies the PCA method and obtains the projection matrix
        :param x: the data matrix on which the PCA is applied
        :return: None

        this function should assign the resulting projection matrix to self.projection_matrix
        """
        # standardized , meaning mean of 0 and a standard deviation of 1
        x_standardized = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        covariance_matrix = np.cov(x_standardized, rowvar=False)  # cov matrix
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        asc_indices = np.argsort(eigenvalues)  # ascending
        desc_indices = asc_indices[::-1]

        top_indices = desc_indices[:self.projection_dim]

        self.projection_matrix = eigenvectors[:, top_indices]  # for each row, get selected features

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        After learning the projection matrix on a given dataset,
        this function uses the learned projection matrix to project new data instances
        :param x: data matrix which the projection is applied on
        :return: transformed (projected) data instances (projected data matrix)
        this function should utilize self.projection_matrix for the operations
        """
        x_standardized = (x - np.mean(x, axis=0)) / np.std(x, axis=0)
        return np.dot(x_standardized, self.projection_matrix)
