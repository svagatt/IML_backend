from pyoselm import OSELMClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import pickle
from pymongo import DESCENDING
import numpy as np

from db_connections import open_database, store_accuracy_results_in_db, store_model_in_db, store_label_encoder_in_db, load_latest_model, get_label_encoder_from_db
from generate_file_hash import get_hash_for_preprocessed_data
from classes import Parameters


async def prepare_test_train_data(parameters, randomstate, hidden_nodes=None, batch_size=None, is_online=None):
    features, labels = await get_features_labels_from_db(parameters, is_online)
    if not is_online:
        le = preprocessing.LabelEncoder()
        le.fit(labels)
        pickled_le = pickle.dumps(le)
        await store_label_encoder_in_db(pickled_le)
    le = await get_label_encoder_from_db()
    depickled_le = pickle.loads(le)
    labels_encoded = depickled_le.transform(labels)
    if is_online:
        return features, labels_encoded
    else:
        X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2,
                                                            random_state=randomstate)
        batches_x = [X_train[:hidden_nodes]] + [X_train[i:i+batch_size] for i in np.arange(hidden_nodes, len(y_train), batch_size)]
        batches_y = [y_train[:hidden_nodes]] + [y_train[i:i+batch_size] for i in np.arange(hidden_nodes, len(y_train), batch_size)]
        return batches_x, batches_y, X_train, X_test, y_train, y_test


async def train_model(randomstate, parameters):
    # the initial phase needs the same amount of samples as the hidden node number for boosting
    # the rest of the data can be split up into different batches
    hidden_nodes: int = 300
    batch_size: int = 50
    initial_batch_test_scores = []
    batches_x, batches_y, X_train, X_test, y_train, y_test = await prepare_test_train_data(parameters, randomstate, hidden_nodes, batch_size, is_online=False)
    model = {'onlineELM': OSELMClassifier(n_hidden=hidden_nodes, activation_func='sigmoid', random_state=randomstate)}
    for name, model in model.items():
        for batch_x, batch_y in zip(batches_x, batches_y):
            # Fit with train data
            model.fit(batch_x, batch_y)
            test_score = model.score(batch_x, batch_y)
            initial_batch_test_scores.append(test_score)
            print(f'Train score: {test_score}')

        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        print(f'Train and test scores for {name}: {train_score}, {test_score}')
        db_elements = Parameters(parameters.subject, parameters.filters, parameters.autoreject, parameters.features,
                                 parameters.channels, train_score, test_score, name)
        await store_accuracy_results_in_db(db_elements, initial_batch_test_scores)
        print('--------Classification done-------')
        trained_model = pickle.dumps(model)
        await store_model_in_db(name, trained_model)


async def train_online(randomstate, parameters):
    X_train, y_train = await prepare_test_train_data(parameters, randomstate, is_online=True)
    model = await load_latest_model()
    # Retrain model
    model.fit(X_train, y_train)
    train_score = model.score(X_train, y_train)

    # Validate scores
    print(f'Train scores for OSELM-batch: {train_score}')
    db_elements = Parameters(parameters.subject, parameters.filters, parameters.autoreject, parameters.features,
                             parameters.channels, train_score, classifier='OSELM-Batch')
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
