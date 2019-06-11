"""
The controller to train and test the pairwise_lstm network
"""

import numpy as np
from keras.models import Model
from keras.models import load_model

from common.clustering.generate_embeddings import generate_embeddings
from common.network_controller import NetworkController
from common.utils.logger import *
from common.utils.paths import *
from common.utils.pickler import load
from .bilstm_2layer_dropout_plus_2dense import bilstm_2layer_dropout
from .core.data_gen import generate_test_data
from .core.pairwise_kl_divergence import pairwise_kl_divergence


class LSTMController(NetworkController):
    def __init__(self):
        super().__init__("pairwise_lstm")
        self.network_file = self.name + "_100"

    def train_network(self):
        bilstm_2layer_dropout(self.network_file, 'speakers_40_clustering_vs_reynolds_train' ,
                              n_hidden1=256, n_hidden2=256, n_classes=100, segment_size=50)

    def get_embeddings(self, cluster_count):
        logger = get_logger('lstm', logging.INFO)
        logger.info('Run pairwise_lstm test\n')


        print("Cluster Count : " + str(cluster_count))

        # Load and prepare train/test data
        #print("old path =============> " + self.get_validation_train_data())
        x_test, speakers_test = load_and_prepare_data(self.get_validation_test_data(), cluster_count=cluster_count * 2)

        # print("old path =============> " + self.get_validation_train_data())
        x_train, speakers_train = load_and_prepare_data(self.get_validation_train_data(), cluster_count=cluster_count * 8)

        # Prepare return values
        set_of_embeddings = []
        set_of_speakers = []
        speaker_numbers = []
        checkpoints = list_all_files(get_experiment_nets(), "*pairwise_lstm*.h5")
        checkpoints = ["pairwise_lstm_100_00999.h5"]

        # Values out of the loop
        metrics = ['accuracy', 'categorical_accuracy', ]
        loss = pairwise_kl_divergence
        custom_objects = {'pairwise_kl_divergence': pairwise_kl_divergence}
        optimizer = 'rmsprop'
        vector_size = 256 * 2

        # Fill return values
        for checkpoint in checkpoints:
            logger.info('Running checkpoint: ' + checkpoint)
            # Load and compile the trained network
            network_file = get_experiment_nets(checkpoint)
            model_full = load_model(network_file, custom_objects=custom_objects)
            model_full.compile(loss=loss, optimizer=optimizer, metrics=metrics)

            # Get a Model with the embedding layer as output and predict
            model_partial = Model(inputs=model_full.input, outputs=model_full.layers[2].output)
            test_output = np.asarray(model_partial.predict(x_test))
            # print("------------------>>>>>>>>>>> test data size\n")
            # print(x_test.shape)
            #
            # print("------------------>>>>>>>>>>> prediction out\n")
            # print(test_output.shape)
            train_output = np.asarray(model_partial.predict(x_train))
            #
            # print("------------------>>>>>>>>>>> train data size\n")
            # print(x_train.shape)
            #
            # print("------------------>>>>>>>>>>> prediction out\n")
            # print(train_output.shape)

            embeddings, speakers, num_embeddings = generate_embeddings(train_output, test_output, speakers_train,
                                                                       speakers_test, vector_size)



            # Fill the embeddings and speakers into the arrays
            set_of_embeddings.append(embeddings)
            set_of_speakers.append(speakers)
            speaker_numbers.append(num_embeddings)

            #return 1

        logger.info('Pairwise_lstm test done.')
        return checkpoints, set_of_embeddings, set_of_speakers, speaker_numbers


def load_and_prepare_data(data_path, segment_size=50, cluster_count=None):
    # Load and generate test data
    x, y, s_list = load(data_path)
    print("stored all speaker utter count ------------------------>>>>>>>>>>>>>>>>>>>>>>")
    print(len(x))
    print("stored all speaker index count ------------------------>>>>>>>>>>>>>>>>>>>>>>")
    print(len(y))

    if cluster_count is not None:
        x = x[0:cluster_count]
        y = y[0:cluster_count]
        s_list = s_list[0:cluster_count]

 #   print(" =============> ", str(len(x)))
  #  print(" =============> ", len(y))
    #print(" =============> ", s_list)
    x, speakers = generate_test_data(x, y, segment_size)

   # print("------------------>>>>>>>>>>> final shape before reshape\n")
   # print(x.shape)

    # Reshape test data because it is an lstm
    return x.reshape(x.shape[0], x.shape[3], x.shape[2]), speakers
