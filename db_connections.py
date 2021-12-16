from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
import sys
import motor.motor_asyncio


def client():
    print('--------Trying to connect to MongoDB--------')
    try:
        mongoclient = motor.motor_asyncio.AsyncIOMotorClient()
        print('Connected successfully')
        return mongoclient
    except ConnectionFailure as e:
        print(f'Could not connect to MongoDB {e}')
        sys.exit(1)


def open_database():
    mongoclient = client()
    db = mongoclient['eeg_data_info']
    return db


def store_accuracy_results_in_db(parameters):
    db = open_database()
    collection = db['results']
    collection.insert_one({
        'subject': parameters.subject,
        'channels': parameters.channels,
        'filters': parameters.filters,
        'autoreject': parameters.autoreject,
        'features': parameters.features,
        'trainScore': parameters.train_score,
        'testScore': parameters.test_score,
        'classifier': parameters.classifier,
        'others': parameters.others,

    })
