from read_save_data_files import get_path, read_raw_fif

from autoreject import AutoReject
import mne


def cut_epochs_by_event_id(event_dict, subject_id, use_autoreject, channel_num):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_{channel_num}_raw.fif'
    raw = read_raw_fif(file_path)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    epochs = mne.Epochs(raw, events, event_dict, -0.1, 2.0, preload=True)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epochs)
    fname = f'/sub_{subject_id}_epo.fif'
    epochs.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)


def cut_epochs_by_event_id_offline(event_dict, subject_id, use_autoreject, channel_num):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_{channel_num}_raw.fif'
    raw = read_raw_fif(file_path)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    epochs = mne.Epochs(raw, events, event_dict, -0.1, 2.0, preload=True)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epochs)
    fname = f'/sub_{subject_id}_epo.fif'
    epochs.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)


def cut_epochs_by_event_id_online(event_dict, subject_id, use_autoreject, channel_num):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_{channel_num}_raw.fif'
    raw = read_raw_fif(file_path)
    """ extract events from raw data """
    raw_events = raw.info['events']
    events = [event['list'].tolist() for event in raw_events]
    epochs = mne.Epochs(raw, events, event_dict, -0.1, 2.0, preload=True)
    # epochs['up'].plot_psd(picks='eeg')
    if use_autoreject is True:
        epochs = use_autoreject_to_remove_noise(epochs)
    fname = f'/sub_{subject_id}_epo.fif'
    epochs.save(get_path('epochs')+fname, overwrite=True, fmt='single', verbose=True)


def use_autoreject_to_remove_noise(epochs):
    """ Auto reject is a paper on automatic artifact removal from EEG data
         https://doi.org/10.1016/j.neuroimage.2017.06.030
    """
    ar = AutoReject()
    epochs_clean = ar.fit_transform(epochs)
    return epochs_clean




