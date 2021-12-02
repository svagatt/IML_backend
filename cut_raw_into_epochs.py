from read_save_data_files import get_path, read_fif_raw
from autoreject import AutoReject

import numpy as np
import mne
import pandas as pd


def cut_epochs_by_event_id(event_dict, subject_id, use_autoreject):
    file_path = get_path('preprocessed_data') + f'/sub_{subject_id}_preprocessed_raw.fif'
    raw = read_fif_raw(file_path)
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




