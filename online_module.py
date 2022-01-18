from read_save_data_files import read_raw_fif, get_path, save_preprocessed_data
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
import elm_classifier
from recorder_connection import get_latest_data_from_buffer


# all the necessary arguments
filter_type = 'cheby2'
features = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
event_dict = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90}
subject_id = 0
auto_reject = False
parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
sample_rate = 500
random_state = 123
is_online = True


def preprocessing_steps(ctr):
    raw = get_latest_data_from_buffer()
    data = preprocessing.preprocess(raw, filter_type)
    save_preprocessed_data(data, subject_id, 64, is_online, ctr)
    cut_raw_into_epochs.cut_epochs_by_event_id_online(subject_id, auto_reject, 64, ctr)
    feature_extraction.extract_features()
    #TODO: online feature extraction
    #TODO: load model to predict
    #TODO: setup data to retrain oselm model



def classify_label(ctr):
    preprocessing_steps(ctr)

