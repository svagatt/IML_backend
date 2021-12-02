from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sys


def client():
    try:
        mongoclient = MongoClient()
        print('--------Trying to connect to MongoDB--------')
        print('Connected successfully')
        return mongoclient
    except ConnectionFailure as e:
        print(f'Could not connect to MongoDB {e}')
        sys.exit(1)


def open_database():
    mongoclient = client()
    db = mongoclient['eeg_data_info']
    return db


def store_accuracy_results_in_db(sub_id, channels, filters, autoreject, feature_extraction_methods, elm_hidden_nodes, activation_method, train_score, test_score):
    db = open_database()
    collection = db['results']
    collection.insert_one({
        'subject': sub_id,
        'channels': channels,
        'filters': filters,
        'autoreject': autoreject,
        'features': feature_extraction_methods,
        'hiddenNodes': elm_hidden_nodes,
        'activationFunction': activation_method,
        'trainScore': train_score,
        'testScore': test_score,
    })
