from pymongo import DESCENDING
from pymongo.errors import ConnectionFailure
import sys
import motor.motor_asyncio
import pickle


sub_id = '2'


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
    cursor = await (db['file_locations'].find().sort({'_id': DESCENDING}).limit(1)).todict()
    return cursor


async def query_for_index():
    cursor = await get_latest_file_location_entry()
    index = cursor['index']
    return index if index is not None else 0


async def save_preprocessed_file_location_in_db(index, subject_id, file_location):
    db = open_database()
    await db['file_locations'].insert_one({
        'index': index,
        'subject': subject_id,
        'preprocessed_data_location': file_location,
    })


async def save_epochs_file_location_in_db(file_location):
    db=open_database()
    index = await query_for_index()
    await db['file_locations'].update_one({'index': index}, {'$set': {'epochs_data_location': file_location}})


async def load_latest_model():
    db = open_database()
    cursor = await (db['trained_models'].find().sort({'_id': DESCENDING}).limit(1)).todict()
    model = pickle.load(cursor['model'])
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


async def get_label_encoder_from_db():
    db = open_database()
    cursor = await (db['label_encoder'].find().sort({'_id': DESCENDING}).limit(1)).todict()
    le = cursor['le']
    return le


async def get_latest_features_from_db(parameters):
    db = open_database()
    features = []
    sub_id = parameters.subject
    filters = parameters.filters
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    cursor = await db[collection_name].find({}, {'features': True}).sort({'_id': DESCENDING}).limit(1)
    async for doc in cursor:
        value = dict(doc)
        features.append(value['features'])
    return features


async def reset_label_in_db(label, parameters):
    db = open_database()
    sub_id = parameters.subject
    filters = parameters.filters
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    cursor = await (db[collection_name].find({}, {'index': True}).sort({'_id': DESCENDING}).limit(1)).todict()
    index = cursor['index']
    await db[collection_name].update_one({'index': index}, {'label': label})
