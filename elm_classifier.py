from pyoselm import ELMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import asyncio

from db_connections import open_database, store_accuracy_results_in_db
from generate_file_hash import get_hash_for_preprocessed_data
from classes import Parameters

db = open_database()


def classify(randomstate, parameters):
    loop = asyncio.get_event_loop()
    loop.run_until_complete(get_features_labels_from_db(parameters))
    # features, labels =
    # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=randomstate)
    #
    # models = {
    #     'elm': ELMClassifier(n_hidden=40, rbf_width=0.2, activation_func='sigmoid', random_state=randomstate),
    #     'SVC': SVC(),
    # }
    #
    # for name, model in models.items():
    #     # Fit with train data
    #     model.fit(X_train, y_train)
    #     train_score = model.score(X_train, y_train)
    #     test_score = model.score(X_test, y_test)
    #
    #     # Validate scores
    #     print(f'Train and test scores for {name}: {train_score}, {test_score}')
    #     print('--------Classification done-------')
    #     db_elements = Parameters(parameters.subject, parameters.filters, parameters.autoreject, parameters.features,
    #                              parameters.channels, train_score, test_score, name)
    #     store_accuracy_results_in_db(db_elements)


async def get_features_labels_from_db(parameters):
    features_list, labels_list = []
    sub_id = parameters.subject
    feature_extraction_methods = parameters.features
    filters = parameters.filters
    channels = parameters.channels
    autoreject = 'autoreject' if parameters.autoreject else ''
    fe_methods = ''
    for method in feature_extraction_methods:
        fe_methods = f'{fe_methods}{method}_'
    collection_name = f'features_{sub_id}_{filters}_{autoreject}_{channels}_{fe_methods}'
    count = await db[collection_name].count_documents({})
    print(count)
    cursor = db[collection_name].find({'hashid': get_hash_for_preprocessed_data(sub_id)}, {'features': True, 'label': True})
    for doc in await cursor.to_list(length=count):
        print(doc)
        # features_list.append(doc['features'])
        # labels_list.append(doc['label'])
    return features_list, labels_list



