from pymongo import DESCENDING, ReturnDocument, ASCENDING
from pymongo.errors import ConnectionFailure
import sys
import motor.motor_asyncio
import pickle
import time

sub_id = '101'


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
    db = mongoclient[f'subject_{sub_id}_eeg_data']
    return db


async def store_accuracy_results_in_db(parameters):
    db = open_database()
    collection = db['results']
    await collection.insert_one({
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


async def get_latest_file_location_entry():
    db = open_database()
    doc = await db['file_locations'].find_one(sort=[('_id', DESCENDING)])
    return doc


async def query_for_index() -> int:
    db = open_database()
    if 'file_locations' not in await db.list_collection_names():
        return 0
    else:
        doc = await get_latest_file_location_entry()
        index = doc['index']
        return index


async def save_preprocessed_file_location_in_db(index, subject_id, file_location):
    db = open_database()
    await db['file_locations'].insert_one({
        'index': index,
        'subject': subject_id,
        'preprocessed_data_location': file_location,
    })


async def save_epochs_file_location_in_db(file_location):
    db = open_database()
    index = await query_for_index()
    await db['file_locations'].update_one({'index': index}, {'$set': {'epochs_data_location': file_location}})


async def get_preprocessed_file_location() -> str:
    db = open_database()
    doc = await db['file_locations'].find_one(sort=[('_id', DESCENDING)], limit=1)
    location = doc['preprocessed_data_location']
    return location


async def load_latest_model() -> bytes:
    db = open_database()
    doc = await db['trained_models'].find_one(sort=[('_id', DESCENDING)], limit=1)
    model = pickle.loads(doc['model'])
    return model


async def store_model_in_db(name, trained_model):
    db = open_database()
    collection_name = 'trained_models'
    await db[collection_name].insert_one({
        'name': name,
        'model': trained_model,
    })


async def store_label_encoder_in_db(le):
    db = open_database()
    collection_name = 'label_encoder'
    await db[collection_name].insert_one({
        'le': le,
    })


async def get_label_encoder_from_db() -> bytes:
    db = open_database()
    doc = await db['label_encoder'].find_one(sort=[('_id', DESCENDING)], limit=1)
    le = doc['le']
    return le


async def get_latest_features_from_db(parameters) -> list:
    db = open_database()
    features = []
    filters = parameters.filters
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    doc = await db[collection_name].find_one(projection={'features': True}, sort=[('_id', DESCENDING)], limit=1)
    features.append(doc['features'])
    return features


async def reset_label_in_db(label, parameters):
    db = open_database()
    filters = parameters.filters
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    _id = await db[collection_name].estimated_document_count()
    result = await db[collection_name].find_one_and_update({'_id': _id}, {'$set': {'label': label}}, return_document=ReturnDocument.AFTER)
    print(result)


def close_database():
    client().close_session()


async def insert_event_into_db(events):
    db = open_database()
    collection_name = 'events'
    await db[collection_name].insert_one({
        'events': events,
    })


async def update_event(event_id):
    db = open_database()
    doc = await db['events'].find_one(projection={'_id': True}, sort=[('_id', DESCENDING)], limit=1)
    index = doc['_id']
    await db['events'].update_one({'_id': index}, {'$set': {'eventId': event_id}})


async def save_time_when_refreshed(name):
    db = open_database()
    await db['recording_lapse'].insert_one({name: time.time()})


async def get_recording_start_time() -> float:
    db = open_database()
    doc = await db['recording_lapse'].find_one(projection={'start_time': True}, sort=[('start_time', DESCENDING)], limit=1)
    return doc['start_time']


async def get_latest_refresh_times() -> float:
    db = open_database()
    latest_refresh_times = []
    doc = await db['recording_lapse'].find_one(projection={'refresh_buffer': True}, sort=[('_id', DESCENDING)], limit=2)
    return doc['refresh_buffer']
