#!/usr/bin/env python
# coding: utf-8
# %%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from time import time
import tensorflow.keras as keras
from sklearn import svm
import numpy as np
from deep_audio import Directory, JSON, Model, Process
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
# %%
method_algo = 'mfcc'
n_rate = 24000
runprocesses = ['perceptron', 'svm']
n_audios = 40
n_segments = 50
# %%
for library in ['melbanks', 'psf']:
    # library = 'melbanks'

    DATASET_PATH = f'processed/{library}_{n_audios}-{n_segments}_{n_rate}.json'

    X, y, mapping = Directory.load_json_data(
        DATASET_PATH)

    sampling_rate = n_rate
    random_state = 42
    # %%
    if 'perceptron' in runprocesses:
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=random_state)

        X_train, X_valid, y_train, y_valid = train_test_split(X_train,
                                                              y_train,
                                                              test_size=0.2,
                                                              stratify=y_train,
                                                              random_state=random_state)

        def build_model():
            # build the network architecture
            model = keras.Sequential([
                # input layer
                keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

                # 1st hidden layer
                keras.layers.Dense(512, activation='relu'),

                # 2nd hidden layer
                keras.layers.Dense(256, activation='relu'),

                # 3rd hidden layer
                keras.layers.Dense(128, activation='relu'),

                # output layer
                keras.layers.Dense(len(mapping), activation='softmax'),
            ])

            optimizer = keras.optimizers.Adam(learning_rate=0.0001)

            model.compile(optimizer=optimizer,
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])

            return model

        kc = KerasClassifier(build_fn=build_model,
                             epochs=2000, batch_size=128, verbose=1, )

        param_grid = {}

        model = GridSearchCV(
            estimator=kc, param_grid=param_grid, n_jobs=-1, cv=3)

        grid_result = model.fit(X_train, y_train)

        score_test = model.score(X_test, y_test)
        score_valid = model.score(X_valid, y_valid)
        score_train = model.score(X_train, y_train)

        Model.dump_grid(
            f'tests/perceptron/{library}_{n_audios}_{n_segments}/info_{Process.pad_accuracy(score_test)}_{time()}.json',
            model=model,
            method='Grid Perceptron',
            sampling_rate=sampling_rate,
            seed=random_state,
            library=library,
            sizes=[len(X_train), len(X_valid), len(X_test)],
            score_train=score_train,
            score_test=score_test,
            score_valid=score_valid
        )
    # %%
    if 'svm' in runprocesses:

        x_holder = []

        for row in X:
            x_holder.append(row.flatten())

        X = np.array(x_holder)

        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            stratify=y,
                                                            random_state=random_state)

        param_grid = {
            'C': [10],
            # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
            'kernel': ['rbf'],
            'decision_function_shape': ['ovo']
        }

        model = GridSearchCV(svm.SVC(), param_grid, cv=3,
                             refit=True, verbose=2, n_jobs=-1)

        model.fit(X_train, y_train)

        score_test = model.score(X_test, y_test)

        score_train = model.score(X_train, y_train)

        Model.dump_grid(
            f'tests/svm/{library}_{n_audios}_{n_segments}/info_{Process.pad_accuracy(score_test)}_{time()}.json',
            model=model,
            method='Grid Perceptron',
            sampling_rate=sampling_rate,
            seed=random_state,
            library=library,
            sizes=[len(X_train), 0, len(X_test)],
            score_train=score_train,
            score_test=score_test,
        )
