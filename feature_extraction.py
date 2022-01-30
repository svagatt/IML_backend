"""[1] Jean-Baptiste SCHIRATTI, Jean-Eudes LE DOUGET, Michel LE VAN QUYEN, Slim ESSID, Alexandre GRAMFORT,
“An ensemble learning approach to detect epileptic seizures from long intracranial EEG recordings” Proc. IEEE ICASSP Conf. 2018
  used for mne feature extraction"""

import mne_features
from pymongo import DESCENDING
import numpy as np
import copy

from read_save_data_files import get_path, read_fif_epochs
from db_connections import open_database, get_latest_file_location_entry, query_for_index
from generate_file_hash import get_hash_for_preprocessed_data
from wavelet_decomposition import Wavelet_Decomposition

wavelet = Wavelet_Decomposition()


def get_epochs_path_offline(sub_id):
    file_path = get_path('epochs') + f'/sub_{sub_id}_epo.fif'
    return file_path


async def get_epochs_path_online():
    doc = await get_latest_file_location_entry()
    file_path = doc['epochs_data_location']
    return file_path


async def extract_features(parameters, samplerate, event_dict, is_online=None):
    db = open_database()
    sub_id = parameters.subject
    channel_num = parameters.channels
    feature_extraction_methods = parameters.features
    feature_list = copy.deepcopy(feature_extraction_methods)
    event_keys = event_dict.keys()
    epochs = None
    epochs_available = True
    if is_online:
        ctr = await query_for_index()
        filehash = get_hash_for_preprocessed_data(sub_id, channel_num, is_online, ctr)
        try:
            epochs = read_fif_epochs(await get_epochs_path_online())
        except AttributeError:
            epochs_available = False
            print('----Empty epoch----')
        event_keys = ['Dummy']
    else:
        filehash = get_hash_for_preprocessed_data(sub_id, channel_num)
        epochs = read_fif_epochs(get_epochs_path_offline(sub_id))
    if epochs_available:
        collection_name = get_collection_name(parameters)
        if collection_name not in await db.list_collection_names():
            print('--------Creating Database------------')
            print('--------Extracting features----------')
            await extract_feature_to_collection(event_keys, epochs, feature_list, samplerate, db, parameters, filehash, is_online)

        else:
            print('------Database already exists--------')
            print('------Adding features------')
            await extract_feature_to_collection(event_keys, epochs, feature_list, samplerate, db, parameters, filehash, is_online)
    return epochs_available


async def extract_feature_to_collection(event_keys, epochs, feature_list, samplerate, db, parameters, filehash, is_online):
    features_wavelet_npy: np.array = []
    for event in event_keys:
        data = epochs.get_data(item=event)
        print(feature_list)
        if 'wavelet_dec' in feature_list:
            features_wavelet_npy = wavelet.feature_vector(data)
            feature_list.remove('wavelet_dec')
        features_mne_npy = mne_features.feature_extraction.extract_features(data, samplerate, feature_list, n_jobs=1)
        features_npy = np.column_stack((features_wavelet_npy, features_mne_npy))
        await create_collection_with_features(parameters, db, features_npy, event, filehash, is_online)
        feature_list.insert(0, 'wavelet_dec')


async def create_collection_with_features(parameters, db, features_npy, label, filehash, is_online):
    collection_name = get_collection_name(parameters)
    feature_collection = db[collection_name]
    index = 0
    if collection_name not in await db.list_collection_names():
        index = 0
    else:
        cursor = feature_collection.find().sort('_id', DESCENDING).limit(1)
        async for doc in cursor:
            index = doc['_id']
    for features in features_npy:
        index += 1
        doc_dict = {'_id': index, 'hashid': filehash,
                    'features': features.tolist(),
                    'label': label,
                    'online': 'True' if is_online is not None else 'False'
                    }
        await insert_doc_in_collection(feature_collection, doc_dict)


async def insert_doc_in_collection(feature_collection, doc):
    await feature_collection.insert_one(doc)


def get_collection_name(parameters) -> str:
    sub_id = parameters.subject
    filters = parameters.filters
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    return collection_name
