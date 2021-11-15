from mne.preprocessing import ICA
components = 25
random = 23
decim = 3
def fast_ica(raw):
    ica = ICA(n_components=components, random_state=random)
    print(ica)
    ica.fit(raw, picks='eeg', decim=decim)
    ica.plot_components()
