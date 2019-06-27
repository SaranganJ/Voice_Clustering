import pickle
from networks.pairwise_lstm.lstm_controller import LSTMController
from common.utils.paths import *
from common.utils.pickler import load

import controller


def storeDict(filename, count):
    with open(get_experiment_results(filename + str(count) + "1"), 'wb') as f:
        print("Saved Dictionary Path")
        print(get_experiment_results(filename + str(count)) + "1")
        pickle.dump(mr_dict, f, -1)


def storeDict2(filename, count):
    with open(get_experiment_results(filename + str(count) + "2"), 'wb') as f:
        print("Saved Dictionary Path")
        print(get_experiment_results(filename + str(count)) + "2")
        pickle.dump(mr_dict2, f, -1)


def storeDict3(filename, count):
    with open(get_experiment_results(filename + str(count) + "3"), 'wb') as f:
        print("Saved Dictionary Path")
        print(get_experiment_results(filename + str(count)) + "3")
        pickle.dump(mr_dict3, f, -1)


def storeDict4(filename, count):
    with open(get_experiment_results(filename + str(count) + "4"), 'wb') as f:
        print("Saved Dictionary Path")
        print(get_experiment_results(filename + str(count)) + "4")
        pickle.dump(mr_dict4, f, -1)


lstm_controller = LSTMController()
lstm_controller.val_data = "VCTK_100_ON_LSTM"
cluster_range = 60
mr_dict = {}
mr_dict2 = {}
mr_dict3 = {}
mr_dict4 = {}

vector_size = [512, 256, 128]

for ii in range(42, cluster_range + 1):

    for vector in vector_size:

        print(
            "#############################################Executing LSTM controller for the vector size################################## :  " + str(
                vector))
        print("\n")

        temp = int(512 / vector)

        for j in range(temp):

            # if j != 7:
            #     continue

            print("Executing LSTM controller for the vector size :  " + str(vector) + " and index " + str(j))
            print("\n")
            lstm_controller.test_network(cluster_count=ii, vector_size=vector, mr_list=mr_dict, mr_list2=mr_dict2,
                                         mr_list3=mr_dict3, mr_list4=mr_dict4, j=j)

            print("................Running CLuster loop for Speaker Count = " + str(ii) + "..................")

            if ii == cluster_range:
                storeDict("lstm_overlap_cluster_timit_40_corpus_", ii)
                storeDict2("lstm_overlap_cluster_timit_40_corpus_", ii)
                storeDict3("lstm_overlap_cluster_timit_40_corpus_", ii)
                storeDict4("lstm_overlap_cluster_timit_40_corpus_", ii)

storeDict("TIMIT_OVERLAP_ON_LSTM_ALL_CHECKPOINT", "ALL")
storeDict2("TIMIT_OVERLAP_ON_LSTM_ALL_CHECKPOINT", "ALL")
storeDict3("TIMIT_OVERLAP_ON_LSTM_ALL_CHECKPOINT", "ALL")
storeDict4("TIMIT_OVERLAP_ON_LSTM_ALL_CHECKPOINT", "ALL")

print("----------->>>>>>>>> From memory")
print(mr_dict)
print("\n")
print(mr_dict2)
print("\n")
print(mr_dict3)
print("\n")
print(mr_dict4)

loaded_dict = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(cluster_range) + "1"))
loaded_dict2 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(cluster_range) + "2"))
loaded_dict3 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(cluster_range) + "3"))
loaded_dict4 = load(get_experiment_results("lstm_overlap_cluster_timit_40_corpus_" + str(cluster_range) + "4"))

print("\n")
print("----------->>>>>>>>> from disk")
print("Hierechial")
print(loaded_dict)

print("\n")
print("K_Means")
print(loaded_dict2)

print("\n")
print("Dominant Sets")
print(loaded_dict3)

print("\n")
print("Vector DS")
print(loaded_dict4)

