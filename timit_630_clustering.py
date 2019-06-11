import pickle
from networks.pairwise_lstm.lstm_controller import LSTMController
from common.utils.paths import *
from common.utils.pickler import load


def storeDict(filename, count):
    with open(get_experiment_results(filename + str(count)), 'wb') as f:
        pickle.dump(mr_dict, f, -1)


lstm_controller = LSTMController()
lstm_controller.val_data = "speakers_all"
cluster_range = 20
mr_dict = {}
for ii in range(20, cluster_range + 1):


    lstm_controller.test_network(cluster_count=ii, mr_list=mr_dict)

    print("................Running CLuster loop for Speaker Count = " + str(ii) + "..................")
    if ii % 10 == 0:
        storeDict("lstm_overlap_cluster_timit_40_corpus_", ii)

storeDict("TIMIT_OVERLAP_ON_LSTM_ALL_CHECKPOINT", "ALL")



print("----------->>>>>>>>> From memory")
for x in mr_dict:
    print(x)
    print(mr_dict[x])

print(mr_dict)





#loaded_dict = load(get_experiment_results("cluster_630_corpus" + str(cluster_range)))




# #p#rint("----------->>>>>>>>> from disk")
# #print("-------->>>>>  mrdict")
# #print(mr_dict)
# #print(loaded_dict)
# for x in loaded_dict:
#     print(loaded_dict[x])
