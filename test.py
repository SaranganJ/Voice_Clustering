from networks.pairwise_lstm.lstm_controller import *
import numpy as np
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load


data_path = "/home/ketharan/ZHAW_deep_voice/common/data/training/speaker_pickles/less_speaker_test_1_2_train.pickle"
# x, speakers = load_and_prepare_data("/home/ketharan/ZHAW_deep_voice/common/data/training"
#             "/speaker_pickles/less_speaker_test_1_2_test.pickle", segment_size=50)
#
# print (x.shape)
# print ("------------------>>>>>>>>>>>\n")
# print(speakers)

# x, y, s_list = load(data_path)
# print("------------------>>>>>>>>>>> shape loaded\n")
# print(x.shape[0])

shapedX, speakers = load_and_prepare_data(data_path, segment_size=50)


#print(shapedX.shape)
#print("------------------>>>>>>>>>>>\n")
#print(speakers)
# print("------------------>>>>>>>>>>>\n")
# print(s_list)