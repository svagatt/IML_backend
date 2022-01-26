from read_save_data_files import get_path, read_raw_fif
from db_connections import query_for_index, save_epochs_file_location_in_db, get_preprocessed_file_location, insert_event_into_db

from autoreject import AutoReject
import mne


def cut_epochs_by_event_id(event_dict, subject_id, use_autoreject, channel_num):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_{channel_num}_raw.fif'
    raw = read_raw_fif(file_path)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    epochs = mne.Epochs(raw, events, event_dict, -0.2, 2.0,  preload=True)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epochs)
    fname = f'/sub_{subject_id}_epo.fif'
    epochs.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)


def cut_epochs_by_event_id_offline(event_dict, subject_id, use_autoreject):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_raw.fif'
    raw = read_raw_fif(file_path)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    print(len(events))
    epochs = mne.Epochs(raw, events, event_dict, -0.2, 1.5, (-0.2, 0), preload=True)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epochs)
    fname = f'/sub_{subject_id}_epo.fif'
    epochs.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)


async def cut_epochs_by_event_id_online(subject_id, use_autoreject, event_dict):

    file_path = await get_preprocessed_file_location()
    raw = read_raw_fif(file_path)
    print(raw)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    await insert_event_into_db(events)
    print(events)
    epoch = mne.Epochs(raw, events, event_dict, -0.1, 0.8, (None, 0.0), preload=True, reject=None, flat=None, reject_by_annotation=False, reject_tmax=None)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epoch)
    index = await query_for_index()
    fname = f'/sub_{subject_id}_{index}_epo.fif'
    epoch.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)
    await save_epochs_file_location_in_db(f"{get_path('epochs')}{fname}")


def use_autoreject_to_remove_noise(epochs):
    """ Auto reject is a paper on automatic artifact removal from EEG data
         https://doi.org/10.1016/j.neuroimage.2017.06.030
    """
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean




