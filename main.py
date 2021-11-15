from preprocessing import preprocess
import numpy as np
import os
import mne

mne.utils.set_config('MNE_USE_CUDA', 'false')
current_number = '2'
filepath = os.getcwd() + '/sample_test_data' + "/participant_" + current_number + '.fif'
raw = mne.io.read_raw_fif(filepath, preload=True).load_data()
raw_data = preprocess(raw)
