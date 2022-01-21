from read_save_data_files import read_raw_fif, get_path, save_preprocessed_data_offline
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
import elm_classifier

import asyncio

# all the necessary arguments
filter_type = 'cheby2'
features = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
event_dict = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90}
subject_id = 111
auto_reject = False
parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
random_state = subject_id

# open the offline recorded data
path = get_path('offline_module_data')
raw_data = read_raw_fif(f'{path}/subject_{subject_id}_raw.fif')
sample_rate = raw_data.info['sfreq']

preprocessed_data = preprocessing.preprocess(raw_data, filter_type)
save_preprocessed_data_offline(preprocessed_data, subject_id, 64)
cut_raw_into_epochs.cut_epochs_by_event_id_offline(event_dict, subject_id, auto_reject, 64)

loop = asyncio.get_event_loop()
loop.run_until_complete(feature_extraction.extract_features(parameters, sample_rate, event_dict))
loop.run_until_complete(elm_classifier.train_model(random_state, parameters))
