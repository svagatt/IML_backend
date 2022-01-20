from read_save_data_files import save_preprocessed_data_online, get_event_dict
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
from elm_classifier import train_online
from recorder_connection import get_latest_data_from_buffer
from db_connections import load_latest_model, get_label_encoder_from_db, get_latest_features_from_db, reset_label_in_db


import pickle


# all the necessary arguments
filter_type = 'cheby2'
features = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
subject_id = 777
# events that occur during the recording
# event_dict = get_event_dict(subject_id)
auto_reject = False
parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
sample_rate = 500
random_state = subject_id
is_online = True


async def preprocessing_steps():
    raw = get_latest_data_from_buffer()
    data = preprocessing.preprocess(raw, filter_type)
    await save_preprocessed_data_online(data, subject_id, 64, is_online)
    await cut_raw_into_epochs.cut_epochs_by_event_id_online(subject_id, auto_reject, 64)
    await feature_extraction.extract_features(parameters, sample_rate, event_dict, True)


async def classify_label():
    await preprocessing_steps()
    data = await get_latest_features_from_db()
    label = await get_label(data)
    return label


def find_event_id_of_label(label):
    for key, value in event_dict:
        if label in key and key==f'{label}_start':
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
    split_label= predicted_label.split('_')
    return split_label[0]


async def set_right_label(label):
    stored_label = ''
    for key, value in event_dict:
        if label in key:
            stored_label = key
    await reset_label_in_db(stored_label, parameters)
