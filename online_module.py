from read_save_data_files import save_preprocessed_data_online, get_path, read_raw_fif, set_montage
import preprocessing
import cut_raw_into_epochs
import feature_extraction
from classes import Parameters
from elm_classifier import train_online
import recorder_connection
from db_connections import sub_id, load_latest_model, get_label_encoder_from_db, get_latest_features_from_db, reset_label_in_db, update_event, store_predicted_label, store_corrected_label


import pickle
import mne
import asyncio
from sklearn.utils.validation import DataConversionWarning

# all the necessary arguments
filter_type: str = 'cheby2'
# features: list = ['wavelet_dec', 'hurst_exp', 'skewness', 'std', 'hjorth_complexity', 'higuchi_fd', 'spect_entropy', 'svd_fisher_info', 'app_entropy', 'pow_freq_bands']
features: list = ['wavelet_dec', 'mean', 'skewness', 'std', 'variance']
subject_id: int = sub_id
# events that occur during the recording
dummy_event_dict = {'Dummy': 99}
event_dict: dict = {'Schraube_start': 10, 'Platine_start': 20, 'Gehäuse_start': 30, 'Werkbank_start': 40, 'Fließband_start': 50, 'Boden_start': 60, 'Lege_start': 70, 'Halte_start': 80, 'Hebe_start': 90}
auto_reject: bool = False
parameters: Parameters = Parameters(subject_id, filter_type, auto_reject, features, 64)
sample_rate: float = 500
random_state: int = subject_id
is_online: bool = True


async def preprocessing_steps():
    data = recorder_connection.get_latest_data_from_buffer()
    print(data)
    shape: tuple = data.shape
    if shape[1] != 0:
        # filtered_data = scipy_filter.filter_data(sample_rate, data)
        info = mne.create_info(ch_names=recorder_connection.get_ch_names(), sfreq=sample_rate, ch_types='eeg')
        info['events'] = [{'list': event} for event in recorder_connection.get_events()[-2:]]
        raw = mne.io.RawArray(data, info)
        # preprocessed_data = mne.io.RawArray(filtered_data, info)

        preprocessed_data = preprocessing.preprocess(raw, filter_type, is_online=is_online)
        await save_preprocessed_data_online(preprocessed_data, subject_id, 64)
        await cut_raw_into_epochs.cut_epochs_by_event_id_online(subject_id, auto_reject, dummy_event_dict)
        epochs_available = await feature_extraction.extract_features(parameters, sample_rate, dummy_event_dict, True)
        return True and epochs_available
    else:
        print('--------Data was not recorded, please try again--------')
        return False


async def classify_label():
    data_available: bool = await preprocessing_steps()
    if not data_available:
        return 'NoData'
    else:
        data = await get_latest_features_from_db(parameters)
        label = await get_label(data)
        return label


async def retrain_model():
    await train_online(random_state, parameters)


async def get_label(data):
    print('-----Loading Model to Predict-----')
    model = await load_latest_model()
    prediction = model.predict(data)
    pickled_le = await get_label_encoder_from_db()
    le = pickle.loads(pickled_le)
    try:
        predicted_label = le.inverse_transform([prediction])[0]
    except DataConversionWarning:
        predicted_label = 'Nicht erkannt_'
    print(f'Predicted Label: {predicted_label}')
    await store_predicted_label(predicted_label)
    split_label = predicted_label.split('_')
    return split_label[0]


async def set_right_label(label):
    for (key, value) in event_dict.items():
        if label in key:
            await store_corrected_label(key)
            await reset_label_in_db(key, parameters)
            await update_event(value)
