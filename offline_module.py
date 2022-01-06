import read_save_data_files
import preprocessing
import cut_raw_into_epochs
import feature_extraction
import elm_classifier

filter_name = 'butter'

# open the offline recorded data
offline_data_path = read_save_data_files.get_path('offline_data')
data = read_save_data_files.read_fif_raw(offline_data_path)
preprocessed_data = preprocessing.preprocess(data, filter_name, 64)
# write to cut the raw data into epochs for the offline module
