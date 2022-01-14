"""[1] Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT,
“An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings” Proc. IEEE ICASSP Conf. 2018
  used for mne feature extraction"""

import mne_features
from pymongo import DESCENDING
import numpy as np
import copy

from read_save_data_files import get_path, read_fif_epochs
from db_connections import open_database
from generate_file_hash import get_hash_for_preprocessed_data
from wavelet_decomposition import Wavelet_Decomposition

wavelet = Wavelet_Decomposition()


def get_epochs_path(sub_id):
    file_path = get_path('epochs') + f'/sub_{sub_id}_epo.fif'
    return file_path


async def extract_features(parameters, samplerate, event_dict):
    sub_id = parameters.subject
    channel_num = parameters.channels
    filehash = get_hash_for_preprocessed_data(sub_id, channel_num)
    feature_extraction_methods = parameters.features
    feature_list = copy.deepcopy(feature_extraction_methods)
    db = open_database()
    epochs = read_fif_epochs(get_epochs_path(sub_id))
    event_keys = event_dict.keys()
    collection_name = get_collection_name(parameters)
    if collection_name not in await db.list_collection_names():
        print('--------Extracting features----------')
        for event in event_keys:
            data = epochs.get_data(item=event)
            if 'wavelet_dec' in feature_list:
                features_wavelet_npy = wavelet.feature_vector(data)
                feature_list.remove('wavelet_dec')
            features_mne_npy = mne_features.feature_extraction.extract_features(data, samplerate, feature_list)
            features_npy = np.column_stack((features_wavelet_npy, features_mne_npy))
            await create_collection_with_features(parameters, db, features_npy, event, filehash)
    else:
        print('------Database already exists--------')


async def create_collection_with_features(parameters, db, features_npy, label, filehash):
    collection_name = get_collection_name(parameters)
    feature_collection = db[collection_name]
    if collection_name not in await db.list_collection_names():
        index = 0
    else:
        cursor = feature_collection.find().sort('_id', DESCENDING).limit(1)
        async for doc in cursor:
            index = doc['_id']
    for features in features_npy:
        index += 1
        doc_dict = {'_id': index, 'hashid': filehash, 'features': features.tolist(), 'label': label}
        await insert_doc_in_collection(feature_collection, doc_dict)


def get_femethods_string(feature_extraction_methods) -> str:
    name = ''
    for method in feature_extraction_methods:
        name = f'{name}{method}_'
    return name


async def insert_doc_in_collection(feature_collection, doc):
    await feature_collection.insert_one(doc)


def get_collection_name(parameters) -> str:
    sub_id = parameters.subject
    feature_extraction_methods = parameters.features
    filters = parameters.filters
    channels = parameters.channels
    autoreject = parameters.autoreject
    fe_methods = get_femethods_string(feature_extraction_methods)
    collection_name = f'features_{sub_id}_{filters}_{autoreject}_{channels}_{fe_methods}'
    return collection_name
