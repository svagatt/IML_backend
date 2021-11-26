from preprocessing import preprocess
from cut_raw_into_epochs import cut_epochs_by_event_id
from db_connections import store_accuracy_results_in_db
import read_save_data_files
import feature_extraction
import elm_classifier
import generate_file_hash

from datetime import datetime
import numpy as np
import os
import mne

""" get current timestamp"""
timestamp = datetime.now()
dbname = 'eeg_data_info'
""" subject info """
subject: str = '2'
event_dict: dict = {'up': 21, 'left': 22, 'right': 23, 'pick': 24, 'push': 25}
random_state = np.random.RandomState(42)

""" Define montage """
montage_path = read_save_data_files.get_path('montage_folder')+'/AC-64.bvef'
montage = mne.channels.read_custom_montage(montage_path, head_size=0.085)
mne.utils.set_config('MNE_USE_CUDA', 'false')

filepath = f'{os.getcwd()}/sample_test_data/participant_{subject}.fif'
filehash = generate_file_hash.sha256sum(filepath)
""" read raw data from the fif file """
raw = read_save_data_files.read_fif_raw(filepath)
samplerate = raw.info['sfreq']
raw.set_montage(montage)
raw.drop_channels(['x_dir', 'y_dir', 'z_dir'])
""" Preprocess the data """
# raw.plot(block=True, scalings='auto')
filter_methods = 'cheby2'
raw_data = preprocess(raw, filter_methods)
# raw_data.plot(block=True, scalings='auto')
""" save preprocessed data into a fif file """
read_save_data_files.preprocessed_data(raw_data, subject)
""" cut raw data into epochs based on the data """
cut_epochs_by_event_id(event_dict, subject, use_autoreject=False)
""" extract relevant features using mne.features"""
feature_extraction_methods = ['kurtosis', 'app_entropy', 'wavelet_coef_energy']
features_npy, labels = feature_extraction.extract_features(subject, samplerate, filehash, feature_extraction_methods)
train_score, test_score = elm_classifier.classify(features_npy, labels, random_state)
store_accuracy_results_in_db(subject, 64, filter_methods, 'yes', feature_extraction_methods, 40, 'sigmoid', train_score, test_score)


