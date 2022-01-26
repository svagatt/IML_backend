import numpy as np
from scipy.signal import butter, sosfilt, cheby2


def filter_data(sampling_frequency: float, data: np.ndarray)-> np.ndarray:
    butter_sos = butter(N=2, Wn=[48.0, 52.0], btype='bandstop', fs=sampling_frequency, output='sos')
    filtered_data = sosfilt(butter_sos, data, axis=-1)
    butter_sos = butter(N=4, Wn=1.0, btype='highpass', fs=sampling_frequency, output='sos')
    filtered_data = sosfilt(butter_sos, filtered_data, axis=-1)
    cheby_sos = cheby2(N=17, rs=60.0, Wn=200.0, btype='lowpass', fs=sampling_frequency, output='sos')
    filtered_data = sosfilt(cheby_sos, filtered_data, axis=-1)
    return filtered_data
