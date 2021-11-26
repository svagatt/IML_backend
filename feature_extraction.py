import mne_features
import mne
import pandas as pd
import generate_file_hash

from read_save_data_files import get_path, read_fif_epochs
from db_connections import open_database


def get_epochs_path(sub_id):
    file_path = get_path('epochs') + f'/sub_{sub_id}_epo.fif'
    return file_path


def extract_features(sub_id, samplerate, filehash, feature_extraction_methods):
    db = open_database()
    epochs = read_fif_epochs(get_epochs_path(sub_id))
    labels = epochs.events[:, -1]
    labellist = labels.tolist()
    create_collection_for_labels(db, sub_id, labellist)
    data = epochs.get_data()
    channel_num = data.shape[1]
    features_npy = mne_features.feature_extraction.extract_features(data, samplerate, feature_extraction_methods)
    create_collection_with_features(db, features_npy, sub_id, channel_num, feature_extraction_methods, filehash)
    return features_npy, labels


def create_collection_with_features(db, features_npy, sub_id, channels, methods, filehash):
    fe_methods = ''
    for method in methods:
        fe_methods = f'{fe_methods}{method}_'
    collection_name = f'features_{sub_id}_{channels}_{fe_methods}'
    if collection_name not in db.list_collection_names():
        feature_collection = db[collection_name]
        for features in features_npy:
            feature_collection.insert_one({
                'hash_id': filehash,
                'features': features.tolist(),
            })


def create_collection_for_labels(db, sub_id, label_list):
    label_collection = db['labels']
    filepath = get_epochs_path(sub_id)
    hash_id = generate_file_hash.sha256sum(filepath)
    label_collection.update_one({'_id': hash_id}, {'$setOnInsert': {
        'subject_id': sub_id,
        'labels': label_list}},
            upsert=True)

# extract_features('2', 500, 'fc0ad2901dc0109a36abf521dbfed16ed93d19370a9527c9b1b22c5b8a4c71bb')
