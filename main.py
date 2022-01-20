from preprocessing import preprocess
from cut_raw_into_epochs import cut_epochs_by_event_id_offline
from feature_extraction import extract_features
from classes import Parameters

import read_save_data_files
import elm_classifier
import generate_file_hash
import asyncio

from datetime import datetime
import numpy as np
import os
import mne
import sys

""" get current timestamp"""
timestamp = datetime.now()
dbname = 'eeg_data_info'

""" subject info, parameters and methods """
subject: str = '17'
event_dict: dict = {'up': 21, 'left': 22, 'right': 23, 'pick': 24, 'push': 25}
random_state = np.random.RandomState(42)
feature_extraction_methods = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info'
                              , 'app_entropy', 'pow_freq_bands']
use_autoreject = False
filter_methods = 'cheby2'
# channels = ['C4', 'FC3', 'FC1', 'F5', 'C3', 'F7', 'FT7', 'CZ', 'C5', 'T7', 'P3']
channels = []
channel_num = len(channels) if len(channels) is not 0 else 64
parameters = Parameters(subject, filter_methods, use_autoreject, feature_extraction_methods, channel_num)

""" Define montage """
montage_path = read_save_data_files.get_path('montage_folder') + '/AC-64.bvef'
montage = mne.channels.read_custom_montage(montage_path, head_size=0.085)
mne.utils.set_config('MNE_USE_CUDA', 'false')

filepath = f'{os.getcwd()}/sample_test_data/participant_{subject}.fif'
""" read raw data from the fif file """
raw = read_save_data_files.read_raw_fif(filepath)
samplerate = raw.info['sfreq']
raw.set_montage(montage)
raw.drop_channels(['x_dir', 'y_dir', 'z_dir'])
""" Preprocess the data """
# raw.plot(block=True, scalings='auto')
raw_data = preprocess(raw, filter_methods, channels)
# raw_data.plot(block=True, scalings='auto')
""" save preprocessed data into a fif file """
read_save_data_files.save_preprocessed_data_offline(raw_data, subject, channel_num)

""" cut raw data into epochs based on the data """
cut_epochs_by_event_id_offline(event_dict, subject, use_autoreject, channel_num)
""" extract relevant features using mne.features"""
loop = asyncio.get_event_loop()
loop.run_until_complete(extract_features(parameters, samplerate, event_dict))
""" Classify """
loop.run_until_complete(elm_classifier.train_model(random_state, parameters))
"""Prevent program from stopping"""
print('-----Press enter to exit------')
if input():
    sys.exit(0)
