import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from theano.gradient import np
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load, save



def get_result_files():

    regex = '*best*.pickle'

    files = list_all_files(get_results(), regex)
    for index, file in enumerate(files):
        files[index] = get_results(file)
    return files

def plot_files2(plot_file_name, files):
    """
    Plots the results stored in the files given and stores them in a file with the given name
    :param plot_file_name: the file name stored in common/data/results
    :param files: a set of full file paths that hold result data
    """
    print("Plot File Name")
    print(plot_file_name)

    loaded_dict = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(630) + "1"))
    loaded_dict2 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(630) + "2"))
    loaded_dict3 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(630) + "3"))


    plot_curves(plot_file_name,loaded_dict,loaded_dict2,loaded_dict3)


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


def plot_curves(plot_file_name,loaded_dict,loaded_dict2,loaded_dict3):
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
    hierach_MR.sort()
    print(hierach_MR)

    for x in loaded_dict2:
        kmeans_mr.append(loaded_dict2[x])
        # print(str(loaded_dict[x]) + " appended to " + str(x))
    print("\n")
    print("Kmeans MR")
    kmeans_mr.sort()
    print(kmeans_mr)

    for x in loaded_dict3:
        ds_mr.append(loaded_dict3[x])
        # print(str(loaded_dict[x]) + " appended to " + str(x))
    print("\n")
    print("DS MR")
    ds_mr.sort()
    print(ds_mr)


    print(num_clusters)
    ks = list(loaded_dict3)
    ks = list(map(int, ks))
    print("\n")
    print("Cluster Count")
    ks.sort()
    print(ks)
    print("Minimum Cluster : " + str(min(ks)))
    print("Maximum CLuster : " + str(max(ks))+ "\n")

    maxc = max(ks)
    minc = min(ks)

    # Get various colors needed to plot
    color_map = plt.get_cmap('gist_rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, 5)]

    # Define number of figures
    fig1 = plt.figure(1)
    fig1.set_size_inches(16, 8)

    # # Define Plots
    # mr_plot = plt.subplot2grid((1, 1), (0, 0), colspan=1)
    # mr_plot.set_ylabel('MR')
    # mr_plot.set_xlabel('Number of clusters')
    # # plt.grid(True)
    plt.title("Plot for DominantSets Clustering for pairwise_lstm")
    plt.axis([(minc-0.1),(maxc+0.1), -0.02,0.2])
    plt.xlabel('Number of clusters')
    plt.ylabel("MR")


    value = [ds_mr, hierach_MR , kmeans_mr]
    # Define curves and their values
    # curves = [[mr_plot, value]]

    algorithm = ["DominantSets_Clustering", "Agglomerative_Hierachial_Clustering",
                 "K_Means_Clustering"]

    # Plot all curves
    for index in range(1):
        ymax = max(value[index])
        xpos = value[index].index(ymax)
        print("xpos " + str(xpos))
        xmax = ks[xpos]

        temp = 'Max Value : ' + str((xmax, ymax))

        # plt.annotate(temp  , xy=(xmax, ymax), xytext=(xmax-50, ymax + 0.05),
        #              arrowprops=dict(facecolor=colors[index], shrink=0.03 ),)
        print("plotting " + algorithm[index])
        label = algorithm[index]
        color = colors[index]
        plt.plot(ks, value[index], color=color, label=label)

        # for plot, value in curves:
        #     # plot.plot(ks,value[index], color=color, label=label)


    # Add legend and save the plot
    fig1.legend()
    fig1.show()
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
    plot_files2("all", get_result_files())

