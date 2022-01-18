from datetime import datetime
import os
import mne

datatype = 'eeg'
extension = '.fif'
timestamp = datetime.now()
timestamp_str = timestamp.strftime("%d-%b-%Y (%H:%M:%S.%f)")


def set_file_name(subject, level, channel_num, ctr=None):
    if ctr is not None:
        filename = f'sub_{subject}_{level}_{channel_num}_{ctr}_raw.fif'
    else:
        filename = f'sub_{subject}_{level}_{channel_num}_raw.fif'
    return filename


def get_path(directory_name):
    path = f'{os.getcwd() }/{directory_name}'
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    return path


def save_preprocessed_data(raw, subject, channel_num, is_online, ctr=None):
    level = 'preprocessed'
    path = get_path('preprocessed_data')
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    if is_online:
        raw.save(os.path.join(path, set_file_name(subject, level, channel_num, ctr)), fmt='single', overwrite=True)
    else:
        raw.save(os.path.join(path, set_file_name(subject, level, channel_num)), fmt='single', overwrite=True)


def read_raw_fif(filepath):
    return mne.io.read_raw_fif(filepath, preload=True).load_data()


def read_fif_epochs(filepath):
    return mne.read_epochs(filepath, preload=True)


