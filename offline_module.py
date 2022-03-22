from read_save_data_files import read_raw_fif, get_path, save_preprocessed_data_offline, set_montage
from preprocessing import preprocess
from cut_raw_into_epochs import cut_epochs_by_event_id_offline
from feature_extraction import extract_features
from elm_classifier import train_model
from classes import Parameters
from db_connections import sub_id


import asyncio

# all the necessary arguments
filter_type: str = 'cheby2'
# features: list = ['wavelet_dec', 'mean', 'skewness', 'std', 'variance', 'hurst_exp', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
features: list = ['wavelet_dec', 'mean', 'skewness', 'std', 'variance', 'pow_freq_bands', 'energy_freq_bands']
event_dict: dict = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90}
subject_id: int = sub_id
auto_reject: bool = False
parameters: Parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
random_state: int = subject_id

# open the offline recorded data
path = get_path('offline_module_data')
raw_data = read_raw_fif(f'{path}/subject_{subject_id}_raw.fif')
sample_rate = raw_data.info['sfreq']
raw_data.set_montage(set_montage())
raw_data.drop_channels(['x_dir', 'y_dir', 'z_dir'])
preprocessed_data = preprocess(raw_data, filter_type)
save_preprocessed_data_offline(preprocessed_data, subject_id)
cut_epochs_by_event_id_offline(event_dict, subject_id, auto_reject)

loop = asyncio.get_event_loop()
loop.run_until_complete(extract_features(parameters, sample_rate, event_dict))
loop.run_until_complete(train_model(random_state, parameters))
