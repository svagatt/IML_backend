""" Authored by Johannes Ihl"""

from pywt import wavedec
import pywt
import numpy as np


class Wavelet_Decomposition:
    def __init__(self):
        pass

    def decompose(self, data, max_level=4, fam='db4'):
        return wavedec(data=data, wavelet=fam, level=max_level)

    # shape [n_words * n_repetitions, channels, n_timestamps]
    def feature_vector(self, data, max_level=4, fam='db4'):
        output = []
        for sample in data:
            feature_vector = []
            wav = self.decompose(sample, max_level=max_level, fam=fam)
            level_helper = max_level
            for level in wav:
                feature_vector.append(level.max())
                feature_vector.append(level.min())
                feature_vector.append(np.average(level))
                feature_vector.append(np.std(level))
                feature_vector.append(self.energy(level, level_helper))
                level_helper -= 1
            output.append(np.array(feature_vector))
        # returns: the data set of the input but compressed:
        # the n (=number_of_electrodes) rows (electrodes) that belong to one stimuli/word is compressed into one,
        # so that there is only one row per label (and not n rows for one label)
        return np.array(output)

    def energy(self, coeffs, k):
        k = min(len(coeffs), k)
        return np.sqrt(np.sum(np.array(coeffs[-k]) ** 2)) / len(coeffs[-k])

