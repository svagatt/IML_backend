from datetime import datetime
import os
import mne
import json
from db_connections import save_preprocessed_file_location_in_db, save_epochs_file_location_in_db, query_for_index

datatype = 'eeg'
extension = '.fif'
timestamp = datetime.now()
timestamp_str = timestamp.strftime("%d-%b-%Y (%H:%M:%S.%f)")


def set_file_name(subject, level, channel_num, ctr=None):
    if ctr is not None:
        filename = f'sub_{subject}_{level}_{ctr}_raw.fif'
    else:
        filename = f'sub_{subject}_{level}_{channel_num}_raw.fif'
    return filename


def get_path(directory_name):
    path = f'{os.getcwd() }/{directory_name}'
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    return path


def save_preprocessed_data_offline(raw, subject, channel_num):
    level = 'preprocessed'
    path = get_path('preprocessed_data')
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    raw.save(os.path.join(path, set_file_name(subject, level, channel_num)), fmt='single', overwrite=True)


async def save_preprocessed_data_online(raw, subject, channel_num):
    level = 'preprocessed'
    path = get_path('preprocessed_data')
    if not os.path.exists(path):
        # Path does not exist yet, create it
        os.makedirs(path)
    index = await query_for_index() + 1
    raw.save(os.path.join(path, set_file_name(subject, level, channel_num, index)), fmt='single', overwrite=False)
    await save_preprocessed_file_location_in_db(index, subject, f'{path}/{set_file_name(subject, level, channel_num, index)}')


def read_raw_fif(filepath):
    return mne.io.read_raw_fif(filepath, preload=True).load_data()


def read_fif_epochs(filepath):
    return mne.read_epochs(filepath, preload=True)


def get_event_dict(subject_id) -> dict:
    file_name = f"{get_path('offline_module_data')}/subject_{subject_id}.json"
    file = open(file_name)
    data = json.load(file)
    return data['Event_Dictionary']


def set_montage():
    montage_path = get_path('montage_folder') + '/AC-64.bvef'
    montage = mne.channels.read_custom_montage(montage_path, head_size=0.085)
    return montage
