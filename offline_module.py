from read_save_data_files import read_raw_fif, get_path, save_preprocessed_data
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
subject_id = 0
auto_reject = False
is_online = False
parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
sample_rate = 500
random_state = 123

# open the offline recorded data
path = get_path('offline_module_data')
raw_data_1 = read_raw_fif(f'{path}/subject_{subject_id}_part_1_raw.fif')
raw_data_2 = read_raw_fif(f'{path}/subject_{subject_id}_part_2_raw.fif')
raw_data_3 = read_raw_fif(f'{path}/subject_{subject_id}_part_3_raw.fif')

data = raw_data_1.copy()
data.append([raw_data_2, raw_data_3], True)
print('DEBUG INFO: Total duration after appending:', data.times.max())

preprocessed_data = preprocessing.preprocess(data, filter_type)
save_preprocessed_data(preprocessed_data, subject_id, 64, is_online)

cut_raw_into_epochs.cut_epochs_by_event_id_offline(event_dict, subject_id, auto_reject, 64)


# async def main():
#     await feature_extraction.extract_features(parameters, sample_rate, event_dict)
#     await elm_classifier.classify_offline(random_state, parameters)
#
# if __name__ == '__main__':
#     asyncio.run(main())
