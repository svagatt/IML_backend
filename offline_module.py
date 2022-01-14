from read_save_data_files import read_raw_fif, get_path, append_raw
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
import elm_classifier

# all the necessary arguments
filter_name = 'butter'
event_dict = {}
subject_id = 0
parameters = Parameters()
sample_rate = 500
randomstate = 123

# open the offline recorded data
offline_data_path = get_path('offline_data')
raw_data_1 = read_raw_fif(f'{offline_data_path}/part_1')
raw_data_2 = read_raw_fif(f'{offline_data_path}/part_2')
raw_data_3 = read_raw_fif(f'{offline_data_path}/part_3')

data = raw_data_1.append([raw_data_2, raw_data_3]).copy()
print('DEBUG INFO: Total duration after appending:', data.times.max())

preprocessed_data = preprocessing.preprocess(data, filter_name, 64)
cut_raw_into_epochs.cut_epochs_by_event_id(event_dict, subject_id, False, 64)
feature_extraction.extract_features(parameters,sample_rate, event_dict)

elm_classifier.classify_offline(randomstate, parameters)
