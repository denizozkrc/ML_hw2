import numpy as np
import math


class Distance:
    @staticmethod
    def calculateCosineDistance(x, y, params_none):
        return 1-((np.dot(x, y))/(np.linalg.norm(x)*np.linalg.norm(y)))

    @staticmethod
    def calculateMinkowskiDistance(x, y, p=2):
        """dim = np.shape(x)[1]
        sum = 0
        for i in range(dim):
            sum += math.pow(abs(x[i]-y[i]), p)
        return math.pow(sum, 1/p)"""
        return np.power(np.sum(np.abs(x - y) ** p), 1 / p)

    @staticmethod
    def calculateMahalanobisDistance(x, y, S_minus_1):
        mah = math.sqrt(np.dot(np.dot(np.transpose(x-y), S_minus_1), x-y))
        return mah
