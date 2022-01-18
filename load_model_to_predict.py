import pickle
from pymongo import DESCENDING

from db_connections import open_database

db = open_database()


async def load_model():
    cursor = await (db['trained_models'].find().sort({'_id': DESCENDING}).limit(1)).todict()
    model = pickle.load(cursor['model'])
    return model


async def get_label():
    model = await load_model()
    label = await model.predict()
    return label
