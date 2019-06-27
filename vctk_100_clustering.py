import pickle
from networks.pairwise_lstm.lstm_controller import LSTMController
from common.utils.paths import *
from common.utils.pickler import load
import controller


lstmcontroller = LSTMController
lstmcontroller.val_data = "speakers_all"
cluster_range = 22
mr_dict = {}
for i in range(20 , cluster_range + 1):
    print("................Running CLuster loop for Speaker Count = " + str(i)+ "..................")
    lstmcontroller.test_network(cluster_count=i, mr_list=mr_dict)


with open(get_experiment_results("VCTK_100_speakers_100_00999" + str(cluster_range)), 'wb') as f:
    pickle.dump(mr_dict, f, -1)

loaded_dict = load(get_experiment_results("VCTK_100_speakers_100_00999" + str(cluster_range)))

#p#rint("----------->>>>>>>>> from disk")
#print("-------->>>>>  mrdict")
print(mr_dict)

