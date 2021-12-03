import mne_features
import numpy as np
from pymongo import ASCENDING, DESCENDING

from read_save_data_files import get_path, read_fif_epochs
from db_connections import open_database
from generate_file_hash import get_hash_for_preprocessed_data


def get_epochs_path(sub_id):
    file_path = get_path('epochs') + f'/sub_{sub_id}_epo.fif'
    return file_path


def extract_features(parameters, samplerate, event_dict):
    sub_id = parameters.subject
    feature_extraction_methods = parameters.features
    db = open_database()
    epochs = read_fif_epochs(get_epochs_path(sub_id))
    event_keys = event_dict.keys()
    for event in event_keys:
        data = epochs.get_data(item=event)
        features_npy = mne_features.feature_extraction.extract_features(data, samplerate, feature_extraction_methods)
        create_collection_with_features(parameters, db, features_npy, event)


def create_collection_with_features(parameters, db, features_npy, label):
    """
    #TODO: add a hash file to compare logs in features collections and have only one features collection
    :param parameters: parameters of the eeg device used for extraction
    :param db: the db instance
    :param features_npy: numpy array with features
    :param label:the label associated with the epochs to add it to the features
    :return:
    """
    sub_id = parameters.subject
    feature_extraction_methods = parameters.features
    filters = parameters.filters
    channels = parameters.channels
    filehash = get_hash_for_preprocessed_data(sub_id)
    autoreject = 'autoreject' if parameters.autoreject else ''
    fe_methods = ''
    for method in feature_extraction_methods:
        fe_methods = f'{fe_methods}{method}_'
    collection_name = f'features_{sub_id}_{filters}_{autoreject}_{channels}_{fe_methods}'
    if collection_name not in db.list_collection_names():
        index = 0
        """ check if features are bring extracted for the same preprocessed hash index"""
    else:
        feature_collection = db[collection_name]
        cursor = feature_collection.find().sort('_id', DESCENDING).limit(1)
        for doc in cursor:
            hashid = doc['hashid']
            index = doc['_id']
        if hashid != get_hash_for_preprocessed_data(sub_id):
            for features in features_npy:
                index += 1
                feature_collection.insert_one({
                    '_id': index,
                    'hashid': filehash,
                    'features': features.tolist(),
                    'label': label,
                })
        else:
            print('---Data already available---')
            return
