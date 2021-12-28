import numpy as np
import mne


def preprocess(raw, filtername, channels):
    sample_rate = raw.info['sfreq']
    print('-----Start Preprocessing-------')
    if channels:
        channel_picks = mne.pick_channels(channels, include=channels)
    else:
        channel_picks = None
    """  https://doi.org/10.3389/fnins.2021.642251
    -> applying a notch filter for power line noise
    -> applying a simple butterworth order 4 highpass and low pass filter
    """
    if filtername == 'butter':
        freqs = np.arange(50, 250, 50)
        raw.notch_filter(freqs, picks=channel_picks)
        raw.filter(1., None)
        raw.filter(None, 50.)
    elif filtername == 'cheby2':
        """
        https://ieeexplore.ieee.org/document/9061628
        """
        pln_filter_params = dict(order=2, ftype='butter')
        raw.notch_filter(50, iir_params=pln_filter_params, picks=channel_picks, method='iir')
        raw.filter(1., None)
        cheby2_filter_params = dict(order=17, ftype='cheby2', output='sos', rs=60.)
        raw.filter(None, 200., iir_params=cheby2_filter_params, method='iir')
    else:
        print('-----Not a valid filter name-----')
    # raw_tmp = raw.copy()
    print('----Preprocessing done----')
    return raw
