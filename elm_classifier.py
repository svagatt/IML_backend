from pyoselm import OSELMClassifier, ELMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from pymongo import DESCENDING

from db_connections import open_database, store_accuracy_results_in_db, store_model_in_db, store_label_encoder_in_db, load_latest_model
from generate_file_hash import get_hash_for_preprocessed_data
from classes import Parameters

le = preprocessing.LabelEncoder()


async def prepare_test_train_data(parameters, randomstate, is_online=None):
    features, labels = await get_features_labels_from_db(parameters, is_online)
    labels_encoded = le.fit_transform(labels)
    pickled_le = pickle.dumps(le)
    await store_label_encoder_in_db(pickled_le)
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=randomstate)
    return X_train, X_test, y_train, y_test


async def train_model(randomstate, parameters):
    X_train, X_test, y_train, y_test = await prepare_test_train_data(parameters, randomstate)
    models = {
        'ELM': ELMClassifier(n_hidden=300, rbf_width=1.0, activation_func='sigmoid', random_state=randomstate),
        'onlineELM': OSELMClassifier(n_hidden=300, activation_func='sigmoid', random_state=randomstate),
    }

    for name, model in models.items():
        # Fit with train data
        model.fit(X_train, y_train)
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(type(model))

        # Validate scores
        print(f'Train and test scores for {name}: {train_score}, {test_score}')
        db_elements = Parameters(parameters.subject, parameters.filters, parameters.autoreject, parameters.features,
                                 parameters.channels, train_score, test_score, name)
        await store_accuracy_results_in_db(db_elements)
        print('--------Classification done-------')
        trained_model = pickle.dumps(model)
        await store_model_in_db(name, trained_model)


async def train_online(randomstate, parameters):
    X_train, X_test, y_train, y_test = await prepare_test_train_data(parameters, randomstate, True)
    model = await load_latest_model()
    # Retrain model
    print(type(model))
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Validate scores
    print(f'Train and test scores for OSELM-Batch: {train_score}, {test_score}')
    db_elements = Parameters(parameters.subject, parameters.filters, parameters.autoreject, parameters.features,
                             parameters.channels, train_score, test_score, 'OSELM-Batch')
    await store_accuracy_results_in_db(db_elements)
    print('--------Classification done-------')
    trained_model = pickle.dumps(model)
    await store_model_in_db('OSELM-Batch', trained_model)


async def get_features_labels_from_db(parameters, is_online):
    features_list, labels_list = [], []
    sub_id = parameters.subject
    filters = parameters.filters
    channels = parameters.channels
    autoreject = parameters.autoreject
    collection_name = f'features_{sub_id}_{filters}_{autoreject}'
    if is_online:
        db = open_database()
        cursor = db[collection_name].find(projection={'features': True, 'label': True}, sort=[('_id', DESCENDING)], limit=9)
        async for doc in cursor:
            value = dict(doc)
            features_list.append(value['features'])
            labels_list.append(value['label'])
        return features_list, labels_list
    else:
        db = open_database()
        cursor = db[collection_name].find({'hashid': get_hash_for_preprocessed_data(sub_id, channels)}, {'features': True, 'label': True})
        async for doc in cursor:
            value = dict(doc)
            features_list.append(value['features'])
            labels_list.append(value['label'])
        return features_list, labels_list
