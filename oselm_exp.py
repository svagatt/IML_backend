import numpy as np
from pyoselm import OSELMClassifier
from sklearn.datasets import fetch_covtype
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time



def prepare_datasets(X, y):
    """Get train and test datasets from data 'X' and 'y',
    with proper standard scaling"""

    idx = [i for i in range(len(y)) if y[i] in [1, 2, 5]]
    X = X[idx, :]
    y = y[idx]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Scale data
    scaler = StandardScaler()
    scaler.fit(X_train, y_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test


def fit_sequential(model, X, y, n_hidden, chunk_size=1):
    """Fit 'model' with data 'X' and 'y', sequentially with mini-batches of
    'chunk_size' (starting with a batch of 'n_hidden' size)"""
    # Sequential learning
    N = len(y)
    # The first batch of data must have the same size as n_hidden to achieve the first phase (boosting)
    batches_x = [X[:n_hidden]] + [X[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
    batches_y = [y[:n_hidden]] + [y[i:i+chunk_size] for i in np.arange(n_hidden, N, chunk_size)]
    i = 1
    for b_x, b_y in zip(batches_x, batches_y):
        if isinstance(model, OSELMClassifier):
            model.fit(b_x, b_y)
            i=i+1
            print(f'{i}Model fit normally')
        else:
            print('model fit partially')
            model.partial_fit(b_x, b_y, classes=[1, 2, 5])

    return model

X, y = fetch_covtype(return_X_y=True)
X_train, X_test, y_train, y_test = prepare_datasets(X, y)

n_hidden = 100

models = {
    "elm": OSELMClassifier(n_hidden=n_hidden, activation_func='sigmoid', random_state=123),
    "sgd": SGDClassifier(random_state=123),
    "par": PassiveAggressiveClassifier(random_state=123),
}

chunk_sizes = [1000]

for name, model in models.items():
    for chunk_size in chunk_sizes:
        print("Chunk size: %i" % chunk_size)

        # Fit with train data
        tic = time.time()
        fit_sequential(model, X_train, y_train, n_hidden, chunk_size)
        toc = time.time()

        # Validate scores
        print("Train score for '%s': %s" % (name, str(model.score(X_train, y_train))))
        print("Test score for '%s': %s" % (name, str(model.score(X_test, y_test))))
        print("Time elapsed: %.3f seconds" % (toc - tic))
        print("")