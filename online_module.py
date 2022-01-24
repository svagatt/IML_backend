from read_save_data_files import save_preprocessed_data_online, get_path, read_raw_fif
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
from elm_classifier import train_online
from recorder_connection import get_latest_data_from_buffer
from db_connections import load_latest_model, get_label_encoder_from_db, get_latest_features_from_db, reset_label_in_db


import pickle
import mne

# all the necessary arguments
filter_type: str = 'cheby2'
features: list = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
subject_id: int = 17
# events that occur during the recording
event_dict: dict = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90}
auto_reject: bool = False
parameters: Parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
sample_rate: float = 500
random_state: int = subject_id
is_online: bool = True


async def preprocessing_steps():
    data = get_latest_data_from_buffer()
    raw = mne.io.RawArray(data, get_mne_info())
    preprocessed_data = preprocessing.preprocess(raw, filter_type)
    await save_preprocessed_data_online(preprocessed_data, subject_id, 64)
    await cut_raw_into_epochs.cut_epochs_by_event_id_online(subject_id, auto_reject)
    await feature_extraction.extract_features(parameters, sample_rate, event_dict, True)


async def classify_label():
    await preprocessing_steps()
    data = await get_latest_features_from_db(parameters)
    label = await get_label(data)
    return label


def find_event_id_of_label(label):
    for key, value in event_dict:
        if label in key and key == f'{label}_start':
            return value
            break


async def retrain_model():
    await train_online()


async def get_label(data):
    model = await load_latest_model()
    prediction = model.predict(data)
    pickled_le = await get_label_encoder_from_db()
    le = pickle.load(pickled_le)
    predicted_label = le.inverse_transform([prediction])[0]
    split_label = predicted_label.split('_')
    return split_label[0]


async def set_right_label(label):
    stored_label = ''
    for key, value in event_dict:
        if label in key:
            stored_label = key
    await reset_label_in_db(stored_label, parameters)


def get_mne_info():
    path = get_path('offline_module_data')
    raw_data = read_raw_fif(f'{path}/subject_{subject_id}_raw.fif')
    return raw_data.info
