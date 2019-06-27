import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import *
from theano.gradient import np
from common.analysis.mr import misclassification_rate
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save


def plot_files(plot_file_name, files):
    """
    Plots the results stored in the files given and stores them in a file with the given name
    :param plot_file_name: the file name stored in common/data/results
    :param files: a set of full file paths that hold result data
    """
    logger = get_logger('analysis', logging.INFO)
    curve_names, set_of_mrs, set_of_homogeneity_scores, \
    set_of_completeness_scores, set_of_number_of_embeddings = read_result_pickle(files)



    for i in range(len(set_of_mrs)):
        print (len(set_of_mrs[i]))


    print("Plot File Name")
    print(plot_file_name)

    loaded_dict = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(40) + "1"))
    loaded_dict2 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(40) + "2"))
    loaded_dict3 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(40) + "3"))


    plot_curves(plot_file_name, curve_names, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores,
                set_of_number_of_embeddings,loaded_dict,loaded_dict2,loaded_dict3)


def read_result_pickle(files):
    """
    Reads the results of a network from these files.
    :param files: can be 1-n files that contain a result.
    :return: curve names, thresholds, mrs, homogeneity scores, completeness scores and number of embeddings
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Read result pickle')
    curve_names = []

    # Initialize result sets
    set_of_thresholds = []
    set_of_mrs = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []
    set_of_number_of_embeddings = []

    logger = get_logger('analysis', logging.INFO)
    # Fill result sets
    for file in files:
        print("File name : " + file)
        curve_name, mrs, homogeneity_scores, completeness_scores, number_of_embeddings = load(file)

        for index, curve_name in enumerate(curve_name):
            set_of_mrs.append(mrs[index])
            set_of_homogeneity_scores.append(homogeneity_scores[index])
            set_of_completeness_scores.append(completeness_scores[index])
            set_of_number_of_embeddings.append(number_of_embeddings[index])
            curve_names.append(curve_name+"_"+file[89:])

    return curve_names, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores, set_of_number_of_embeddings


def plot_curves(plot_file_name, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings,loaded_dict,loaded_dict2,loaded_dict3):
    """
    Plots all specified curves and saves the plot into a file.
    :param plot_file_name: String value of save file name
    :param curve_names: Set of names used in legend to describe this curve
    :param mrs: 2D Matrix, each row describes one dataset of misclassification rates for a curve
    :param homogeneity_scores: 2D Matrix, each row describes one dataset of homogeneity scores for a curve
    :param completeness_scores: 2D Matrix, each row describes one dataset of completeness scores for a curve
    :param number_of_embeddings: set of integers, each integer describes how many embeddings is in this curve
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Plot results')
    logger.info(plot_file_name)
    min_mrs = []
    for mr in mrs:
        min_mrs.append(np.min(mr))


    num_clusters=[]
    hierach_MR = []
    kmeans_mr = []
    ds_mr = []

    for x in loaded_dict:
        num_clusters.append(x)
        hierach_MR.append(loaded_dict[x])
        # print(str(loaded_dict[x]) + " appended to " + str(x))
    print("\n")
    print("Hierachial MR")
    print(hierach_MR)

    for x in loaded_dict2:
        kmeans_mr.append(loaded_dict2[x])
        # print(str(loaded_dict[x]) + " appended to " + str(x))
    print("\n")
    print("Kmeans MR")
    print(kmeans_mr)

    for x in loaded_dict3:
        ds_mr.append(loaded_dict3[x])
        # print(str(loaded_dict[x]) + " appended to " + str(x))
    print("\n")
    print("DS MR")
    print(ds_mr)


    ks = list(loaded_dict)
    ks = list(map(int, ks))
    print("\n")
    print("Cluster Count")
    print(ks)
    print("Minimum Cluster : " + str(min(ks)))
    print("Maximum CLuster : " + str(max(ks))+ "\n")

    maxc = max(ks)
    minc = min(ks)

    # x = zip(*sorted(zip(min_mrs, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings)))
    #
    #
    # print("\n")
    # print(tuple(x))
    # print("\n")
    #
    # min_mrs, curve_names, mrs, homogeneity_scores, completeness_scores, number_of_embeddings = \
    #     (list(t) for t in x)


    # How many lines to plot
    number_of_lines = len(curve_names)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, number_of_lines)]

    # Define number of figures
    fig1 = plt.figure(1)
    fig1.set_size_inches(16, 8)

    # Define Plots
    mr_plot = plt.subplot2grid((2, 3), (0, 0), colspan=2)
    mr_plot.set_ylabel('MR')
    mr_plot.set_xlabel('Number of clusters')
    plt.grid(True)
    plt.axis([minc,maxc, -0.02,1.2])


    # print(mrs)
    #
    #
    # for i in range(1,len(mrs),2):
    #     print(mrs[i])
    #     print(mrs[i][0])
    #     kmeans_mr.append(mrs[i][0])
    #     print(str(mrs[i][0]) + " appended "

    completeness_scores_plot = add_cluster_subplot(fig1, 234, 'completeness_scores')
    homogeneity_scores_plot = add_cluster_subplot(fig1, 235, 'homogeneity_scores')

    value = [hierach_MR , kmeans_mr, ds_mr]
    # Define curves and their values
    curves = [[mr_plot, value]]

    algorithm = ["Agglomerative_Hierachial_Clustering",
                 "K_Means_Clustering",
                 "DominantSets_Clustering"]


    # Plot all curves
    for index in range(3):
        label = algorithm[index]
        color = colors[index]
        # number_of_clusters = np.arange(number_of_embeddings[index], 0, -1)


        for plot, value in curves:
            print(value[index])
            plot.plot(ks,value[index], color=color, label=label)


    # Add legend and save the plot
    fig1.legend()
    # fig1.show()
    fig1.savefig(get_result_png(plot_file_name))
    print("Plot File saved in " + get_result_png(plot_file_name) )
    fig1.savefig(get_result_png(plot_file_name + '.svg'), format='svg')


def add_cluster_subplot(fig, position, y_lable):
    """
    Adds a cluster subplot to the given figure.

    :param fig: the figure which gets a new subplot
    :param position: the position of this subplot
    :param title: the title of the subplot
    :return: the subplot itself
    """
    subplot = fig.add_subplot(position)
    subplot.set_ylabel(y_lable)
    subplot.set_xlabel('number of clusters')
    return subplot


def analyse_results(network_name, checkpoint_names, set_of_predicted_clusters, set_of_true_clusters, embedding_numbers,algorithm, vector, j,
                    cluster_count=None, mr_list=None ,mr_list2=None):
    """
    Analyses each checkpoint with the values of set_of_predicted_clusters and set_of_true_clusters.
    After the analysis the result are stored in the Pickle network_name.pickle and the best Result
    according to min MR is stored in network_name_best.pickle.

    :param network_name: The name for the result pickle.
    :param checkpoint_names: A list of names from the checkpoints. Later used as curvenames,
    :param set_of_predicted_clusters: A 2D array of the predicted Clusters from the Network. [checkpoint, clusters]
    :param set_of_true_clusters: A 2d array of the validation clusters. [checkpoint, validation-clusters]
    :param embeddings_numbers: A list which represent the number of embeddings in each checkpoint.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Run analysis')
    set_of_mrs = []
    set_of_homogeneity_scores = []
    set_of_completeness_scores = []

    print("\nAnalyzing Results for " + algorithm+ "\n")

    print("Checkpoint Names")
    print(checkpoint_names)
    print("\n")

    # print("Set of predicted clusters in Analysis\n")
    # print(set_of_predicted_clusters)
    # print("\n")
    #
    # print("Set of true clusters in Analysis\n")
    # print(set_of_true_clusters)
    # print("\n")

    for index, predicted_clusters in enumerate(set_of_predicted_clusters):
        print(index)

        if checkpoint_names is not None:
            logger.info('Analysing checkpoint:' + checkpoint_names[index-1])

        # print("Predicted clusters in Analysis\n")
        # print(predicted_clusters)
        # print("\n")

        mrs, homogeneity_scores, completeness_scores, mr_list = calculate_analysis_values(predicted_clusters,
                                                                                 set_of_true_clusters[index-1], algorithm ,vector,j ,
                                                                             cluster_count, mr_list, mr_list2)

        set_of_mrs.append(mrs)
        set_of_homogeneity_scores.append(homogeneity_scores)
        set_of_completeness_scores.append(completeness_scores)

    write_result_pickle(network_name, checkpoint_names, set_of_mrs, set_of_homogeneity_scores,
                        set_of_completeness_scores, embedding_numbers, algorithm)
    save_best_results(network_name, checkpoint_names, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, embedding_numbers,algorithm)
    logger.info('Analysis done')


def calculate_analysis_values(predicted_clusters, true_cluster, alorithm, vector,j , cluster_count=None, mr_list=None , mr_list2=None):
    """
    Calculates the analysis values out of the predicted_clusters.

    :param predicted_clusters: The predicted Clusters of the Network.
    :param true_clusters: The validation clusters
    :return: misclassification rate, homogeneity Score, completeness score and the thresholds.
    """
    logger = get_logger('analysis', logging.INFO)
    logger.info('Calculate scores')
    #
    # print("------------------>>>>>>>>>>>  before incremental true clusters\n")
    # print(true_cluster)
    # for i in range(len(true_cluster)):
    #     true_cluster[i] += 1
    #
    # print("------------------>>>>>>>>>>>  after incremental true clusters\n")
    # print(true_cluster)

    # Initialize output
    mrs = np.ones(len(true_cluster))
    homogeneity_scores = np.ones(len(true_cluster))
    completeness_scores = np.ones(len(true_cluster))

    # Loop over all possible clustering
    for i, predicted_cluster in enumerate(predicted_clusters):
        # Calculate different analysis's
        # print(i)
        # print("\n")
        # print(predicted_cluster)

        if alorithm == "K_Means_Clustering":

            print("True Clusters")
            print(true_cluster)
            print("\n")
            print("Predicted Clusters")
            print(predicted_clusters)
            print("\n")

            mrs[i] = misclassification_rate(true_cluster, predicted_cluster)
            print("...................MR value of K means................................\n")
            print(mrs[i])
            print("\n")
            if cluster_count is not None:

                temp = str(cluster_count) + "_" + str(vector) + "_" + str(j)
                mr_list2[temp] = mrs[i]
                print(str(mrs[i]) + " added to mr_list2\n")

        if alorithm == "Agglomerative_Hierachial_Clustering":
            mrs[i] = misclassification_rate(true_cluster, predicted_cluster)
            print(mrs[i])

            if cluster_count is not None and (max(predicted_cluster) == cluster_count):
                temp = str(cluster_count) + "_" + str(vector) + "_" + str(j)
                mr_list[temp] = mrs[i]
                print("...................MR value of Hirechial Clustering................................\n")
                print(mrs[i])
                print("\n")
                print(str(mrs[i]) + " added to mr_list\n")

        homogeneity_scores[i] = homogeneity_score(true_cluster, predicted_cluster)
        completeness_scores[i] = completeness_score(true_cluster, predicted_cluster)

    return mrs, homogeneity_scores, completeness_scores ,mr_list


def save_best_results(network_name, checkpoint_names, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, speaker_numbers,algorithm):
    if len(set_of_mrs) == 1:
        write_result_pickle(network_name + "_best", checkpoint_names, set_of_mrs,
                            set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers,algorithm)
    else:

        # Find best result (min MR)
        min_mrs = []
        for mrs in set_of_mrs:
            min_mrs.append(np.min(mrs))

        min_mr_over_all = min(min_mrs)

        best_checkpoint_name = []
        set_of_best_mrs = []
        set_of_best_homogeneity_scores = []
        set_of_best_completeness_scores = []
        best_speaker_numbers = []
        for index, min_mr in enumerate(min_mrs):
            if min_mr == min_mr_over_all:
                index = index - 1
                best_checkpoint_name.append(checkpoint_names[index])
                set_of_best_mrs.append(set_of_mrs[index])
                set_of_best_homogeneity_scores.append(set_of_homogeneity_scores[index])
                set_of_best_completeness_scores.append(set_of_completeness_scores[index])
                best_speaker_numbers.append(speaker_numbers[index])

        write_result_pickle(network_name + "_best", best_checkpoint_name, set_of_best_mrs,
                            set_of_best_homogeneity_scores, set_of_best_completeness_scores, best_speaker_numbers)


def write_result_pickle(network_name, checkpoint_names, set_of_mrs, set_of_homogeneity_scores,
                        set_of_completeness_scores, number_of_embeddings ,algorithm):
    logger = get_logger('analysis', logging.INFO)

    save((checkpoint_names, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores,
          number_of_embeddings), (get_result_pickle(network_name+"_"+algorithm)))
    logger.info('Write result pickle to ' + str((get_result_pickle(network_name+"_"+algorithm))))


def read_and_safe_best_results():
    checkpoint_names, set_of_mrs, set_of_homogeneity_scores, set_of_completeness_scores, speaker_numbers = read_result_pickle(
        [get_result_pickle('flow_me')])
    save_best_results('flow_me', checkpoint_names, set_of_mrs, set_of_homogeneity_scores,
                      set_of_completeness_scores, speaker_numbers)


if __name__ == '__main__':
    read_and_safe_best_results()
