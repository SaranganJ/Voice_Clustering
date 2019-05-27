import numpy as np
import librosa.display
from scipy.cluster import hierarchy
import random
from matplotlib import pyplot as plot

from common.spectogram.spectrogram_converter import *
from common.utils.paths import *



def save_spectrogramm_png(path):
    # Load the mel spectrogram
    spectrogram = mel_spectrogram(path)

    # Begin the plot
    figure = plot.figure(1)
    plot.imshow(spectrogram[:, 20:160])

    # Add the color bar
    color_bar = plot.colorbar()
    n = np.linspace(0, 35, num=11)
    labels = []
    for l in n:
        labels.append(str(l) + ' dB')
    color_bar.ax.set_yticklabels(labels)

    # Add x and y labels
    plot.xlabel('Spektra (in Zeit)')
    plot.ylabel('Frequenz-Datenpunkte')

    # Save the figure to disc
    figure.savefig(get_result_png('spectrogram'))

def save_spectrogram(spectrogram):
    # Load the mel spectrogram
    suff = random.randint(1, 1000)

    print("------------------>>>>>>>>>>>  Random suffix\n")
    print(suff)
    # Begin the plot
    figure = plot.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.power_to_db(spectrogram,
        ref = np.max), y_axis = 'mel', fmax = 8000,x_axis = 'time')
#    figure.colorbar(format='%+2.0f dB')
    figure.tight_layout()

    figure.savefig(get_result_png('spectrogram' + str(suff)))

    #print("Saved in " + get_result_png('spectrogram'))

def save_create_dendrogram(linkage_matrix):
    figure = plot.figure(figsize=(10, 4))
    dn = hierarchy.dendrogram(linkage_matrix)
    figure.savefig(get_result_png('dendrogram'))




if __name__ == '__main__':
    save_spectrogramm_png('SA1_RIFF.wav')
