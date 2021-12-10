from datetime import datetime
import os
import mne

datatype = 'eeg'
extension = '.fif'
timestamp = datetime.now()
timestamp_str = timestamp.strftime("%d-%b-%Y (%H:%M:%S.%f)")


def set_file_name(subject, level, channel_num):
    filename = f'sub_{subject}_{level}_{channel_num}_raw.fif'
    return filename


def get_path(directory_name):
    path = f'{os.getcwd() }/{directory_name}'
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    return path


def preprocessed_data(raw, subject, channel_num):
    level = 'preprocessed'
    path = get_path('preprocessed_data')
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    raw.save(os.path.join(path, set_file_name(subject, level, channel_num)), fmt='single', overwrite=True)


def read_fif_raw(filepath):
    return mne.io.read_raw_fif(filepath, preload=True).load_data()


def read_fif_epochs(filepath):
    return mne.read_epochs(filepath, preload=True)

