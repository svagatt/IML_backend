from pyoselm import ELMClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from db_connections import open_database


def split_data_into_train_test(collection_name):
    db = open_database()
    db.collection_name


def classify(features_npy, labels, randomstate):
    X_train, X_test, y_train, y_test = train_test_split(features_npy, labels, test_size=0.2, random_state=randomstate)

    models = {
        'elm': ELMClassifier(n_hidden=40, rbf_width=0.2, activation_func='sigmoid', random_state=randomstate),
    }

    for name, model in models.items():
        # Fit with train data
        model.fit(X_train, y_train)

        # Validate scores
        print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
        print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
        print('--------Classification done-------')
        return str(model.score(X_train, y_train)), str(model.score(X_test, y_test))