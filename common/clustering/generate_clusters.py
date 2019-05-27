from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist
from common.spectogram.spectrogram_png_safer import save_create_dendrogram
from common.utils.logger import *


def cluster_embeddings(set_of_embeddings, metric='cosine', method='complete'):
    """
    Calculates the distance and the linkage matrix for these embeddings.

    :param set_of_embeddings: The embeddings we want to calculate on
    :param metric: The metric used for the distance and linkage
    :param method: The linkage method used.
    :return: The embedding Distance and the embedding linkage
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Cluster embeddings')

    set_predicted_clusters = []
    #print("------------------>>>>>>>>>>> det embeddings list\n")
    #print(len(set_of_embeddings))

    for embeddings in set_of_embeddings:
        embeddings_distance = cdist(embeddings, embeddings, metric)

        #print("------------------>>>>>>>>>>> distance matrix\n")
        #print(embeddings_distance.shape)

        embeddings_linkage = linkage(embeddings_distance, method, metric)

        save_create_dendrogram(embeddings_linkage)

        #print("------------------>>>>>>>>>>> threshold list creation\n")
        #print(embeddings_linkage.shape)

        thresholds = embeddings_linkage[:, 2]

        #print("------------------>>>>>>>>>>> threshold list\n")
        #print(thresholds)

        predicted_clusters = []

        for threshold in thresholds:
            predicted_cluster = fcluster(embeddings_linkage, threshold, 'distance')
            predicted_clusters.append(predicted_cluster)

            if (max(predicted_cluster) == 5):
                print("------------------>>>>>>>>>>> predicted cluster\n")
                print(predicted_cluster)

        set_predicted_clusters.append(predicted_clusters)

    return set_predicted_clusters
