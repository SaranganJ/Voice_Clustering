from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans

from common.spectogram.spectrogram_png_safer import save_create_dendrogram
from common.utils.logger import *
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from common.clustering import dominantset as ds
import common.clustering.extractor
import numpy as np


def cluster_embeddings(set_of_embeddings, set_of_true_clusters, algorithm, metric='cosine', method='complete', ):
    """
    Calculates the distance and the linkage matrix for these embeddings.

    :param set_of_embeddings: The embeddings we want to calculate on
    :param metric: The metric used for the distance and linkage
    :param method: The linkage method used.
    :param set_of_true_clusters: The speaker list
    :return: The embedding Distance and the embedding linkage
    """
    logger = get_logger('analysis', logging.INFO)
    print("\n")
    logger.info('Cluster embeddings(Inside Generate Cluster Class)')
    print("\n")

    # print("------------------>>>>>>>>>>> det embeddings list\n")

    ["Agglomerative_Hierachial_Clustering",
     "K_Means_Clustering",
     "DominantSets_Clustering"]
    # print(len(set_of_embeddings))

    for embeddings in set_of_embeddings:

        # print("Dimension of Embeddings")
        # print(embeddings.shape)
        # print("\n")

        if algorithm == "Agglomerative_Hierachial_Clustering":

            set_predicted_clusters = []

            # distance matrix is returned. For each i and j dist(u=XA[i], v=XB[j])`` is computed and stored
            embeddings_distance = cdist(embeddings, embeddings, metric)

            print("Dimension of Distance Matrix")
            print(embeddings_distance.shape)
            print("\n")

            # The hierarchical clustering encoded as a linkage matrix.
            embeddings_linkage = linkage(embeddings_distance, method, metric)
            save_create_dendrogram(embeddings_linkage)

            print("Dimension of hierarchical clustering encoded as a linkage matrix")
            print(embeddings_linkage.shape)
            print("\n")

            thresholds = embeddings_linkage[:, 2]

            print("------------------>>>>>>>>>>> Threshold list\n")
            print(thresholds)

            predicted_clusters = []

            for threshold in thresholds:
                predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
                # print("\n")
                # print(str(predicted_cluster)+ " predicted")
                predicted_clusters.append(predicted_cluster)

                if (max(predicted_cluster) == 5):
                    print("------------------>>>>>>>>>>> predicted cluster when k=5\n")
                    print(predicted_cluster)

            set_predicted_clusters.append(predicted_clusters)

        elif algorithm == "DominantSets_Clustering":

            set_predicted_clusters = []
            predicted_clusters = []

            print("Set of True Clusters[0]")
            a = np.asarray(set_of_true_clusters[0])
            print(a)
            print("\n")

            dos = ds.DominantSetClustering(feature_vectors=embeddings, speaker_ids=a,
                                           metric='cosine', dominant_search=False,
                                           epsilon=1e-6, cutoff=-0.1)

            dos.apply_clustering()
            mr, ari, acp = dos.evaluate()

            print("MR\t\tARI\t\tACP")
            print("{0:.4f}\t\t{1:.4f}\t\t{2:.4f}".format(mr, ari, acp))  # MR - ARI - ACP

        elif algorithm == "K_Means_Clustering":

            set_predicted_clusters = []

            predicted_clusters = []

            for i in range(1):
                k = 20
                kmeans_model = KMeans(n_clusters=k).fit(embeddings)

                # Predicting the clusters
                labels = kmeans_model.predict(embeddings)
                print("---------Predicted Clusters when" + " K =" + str(k) + "-----------------")
                print(labels)
                print("\n")
                predicted_clusters.append(labels)

                # Getting the cluster centers
                # C = kmeans_model.cluster_centers_
                # print("Cluster Centers")
                # print(C)
                # print("\n")

            set_predicted_clusters.append(predicted_clusters)

        else:
            print("The clustering should be Agglo or K-Means or DS")

    return set_predicted_clusters
