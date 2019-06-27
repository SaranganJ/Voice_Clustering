import numpy as np

from common.utils.logger import *


def generate_embeddings(train_output, test_output, train_speakers, test_speakers, vector_size,j):
    """
    Combines the utterances of the speakers in the train- and testing-set and combines them into embeddings.
    :param train_output: The training output (8 sentences)
    :param test_output:  The testing output (2 sentences)
    :param train_speakers: The speakers used in training
    :param test_speakers: The speakers used in testing
    :param vector_size: The size which the output will have
    :return: embeddings, the speakers and the number of embeddings
    """
    logger = get_logger('clustering', logging.INFO)
    logger.info('Generate embeddings')
    num_speakers = len(set(test_speakers))

    # Prepare return variable
    number_embeddings = 2 * num_speakers
    embeddings = []
    speakers = []

    # Create utterances
    embeddings_train, speakers_train = create_utterances(num_speakers, vector_size, train_output, train_speakers,j)
    embeddings_test, speakers_test = create_utterances(num_speakers, vector_size, test_output, test_speakers,j)

    # Merge utterances
    embeddings.extend(embeddings_train)
    embeddings.extend(embeddings_test)
    speakers.extend(speakers_train)
    speakers.extend(speakers_test)

    # print(print("------------------>>>>>>>>>>> embeddings and speakers\n"))
    # print(embeddings)
    # print(speakers)

    return embeddings, speakers, number_embeddings


def multiple(m, n):
    # inserts all elements from n to
    # (m * n)+1 incremented by n.
    a = range(n, (m * n) + 1, n)

    print(*a)


# num_speakers, vector_size, train_output, train_speakers
def create_utterances(num_speakers, vector_size, vectors, y,j):
    """
    Creates one utterance for each speaker in the vectors.
    :param num_speakers: Number of distinct speakers in this vector
    :param vector_size: Number of data in utterance
    :param vectors: The unordered speaker data
    :param y: An array that tells which speaker (number) is in which place of the vectors array
    :return: the embeddings per speaker and the speakers (numbers)
    """

    # Prepare return variables
    embeddings = np.zeros((num_speakers, vector_size))
    speakers = set(y)

    print("utterence shape")
    print(embeddings[0].shape)
    print("\n")

    temp = vector_size / 512
    print("temp")
    print(temp)
    print("\n")

    # Fill embeddings with utterances
    for i in range(num_speakers):

        # Fetch correct utterance
        utterance = embeddings[i]

        # Fetch values where same speaker and add to utterance
        indices = np.where(y == i)[0]

        # Take the vectors output from vectors with the corresponding index
        # Dimension will be (indices,512)
        outputs = np.take(vectors, indices, axis=0)


        for value in outputs:

            if temp == 0.5:

                index_list = [k for k in range(j, 512, 2)]
                value = [value[i] for i in index_list]
                value = np.asarray(value)

            elif temp == 0.25:
                index_list = [k for k in range(j, 512, 4)]
                value = [value[i] for i in index_list]
            elif temp == 0.125:
                index_list = [k for k in range(j, 512, 8)]
                value = [value[i] for i in index_list]
            else:
                s=0

            utterance = np.add(utterance, value)

        # Add filled utterance to embeddings
        embeddings[i] = np.divide(utterance, len(outputs))

    return embeddings, speakers
