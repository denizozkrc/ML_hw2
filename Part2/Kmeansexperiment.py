import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# The datasets are already preprocessed...
dataset1 = pickle.load(open("../datasets/part2_dataset_1.data", "rb"))
dataset2 = pickle.load(open("../datasets/part2_dataset_2.data", "rb"))


def myKMeansFunction(dataset):
    avg = np.zeros(11)
    avg[0] = avg[1] = np.inf
    sil_avg = np.zeros(11)
    sil_avg[0] = sil_avg[1] = np.NINF
    for k_val in range(2, 11):
        total = 0
        sil_total = 0
        for repeat_and_average in range(0, 10):
            smallest = np.inf
            sil_biggest = -2
            for repeat_and_lowest in range(0, 10):
                kmeans = KMeans(n_clusters=k_val, random_state=0)
                prediction = kmeans.fit_predict(dataset)
                sil_score = silhouette_score(dataset, prediction)
                if kmeans.inertia_ < smallest:
                    smallest = kmeans.inertia_
                if sil_score > sil_biggest:
                    sil_biggest = sil_score
            total += smallest
            sil_total += sil_biggest
        avg[k_val] = total/10
        sil_avg[k_val] = sil_total/10
    """best_k = np.argmin(avg)
    sil_best_k = np.argmax(sil_avg)
    print(best_k, sil_best_k)
    print(avg[best_k], sil_avg[sil_best_k])"""

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(0, 11), avg, marker='o', label='Average Inertia')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Inertia')
    plt.title('Inertia vs. Number of Clusters')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(0, 11), sil_avg, marker='o', label='Average Silhouette Score', color='orange')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()


myKMeansFunction(dataset1)

myKMeansFunction(dataset2)
