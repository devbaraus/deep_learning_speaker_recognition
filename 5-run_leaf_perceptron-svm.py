#!/usr/bin/env python
# coding: utf-8

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
import tensorflow.keras as keras
from sklearn import svm
import numpy as np
from deep_audio import Directory, JSON, NumpyEncoder
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier


method_algo = 'mfcc'
n_rate = 24000

runprocesses = ['perceptron', 'svm']

#['leaf', 'melbanks', 'tfbanks', 'sincnet', 'sincnetplus']
for library in ['melbanks', 'tfbanks', 'sincnet', 'sincnetplus']:

    print(f"\n\n\n------------- {library} -------------\n\n\n")

    DATASET_PATH = f'processed/leaf/{library}_{n_rate}.json'

    X, y, mapping = Directory.load_json_data(
        DATASET_PATH, inputs_fieldname=method_algo)

    random_state = 42

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
                             epochs=2000, batch_size=128, verbose=2)

        param_grid = {}

        model = GridSearchCV(
            estimator=kc, param_grid=param_grid, n_jobs=-1, cv=5)

        grid_result = model.fit(X_train, y_train)

        best_params = model.best_params_
        sampling_rate = n_rate

        score_test = model.score(X_test, y_test)
        score_valid = model.score(X_valid, y_valid)
        score_train = model.score(X_train, y_train)

        y_hat = model.predict(X_test)

        dump_info = {
            'method': 'Perceptron',
            'seed': random_state,
            'library': library,
            'feature_method': method_algo,
            'sample_rate': sampling_rate,
            'train_test': [len(X_train), len(X_valid), len(X_test)],
            'score_train': score_train,
            'score_valid': score_valid,
            'score_test': score_test,
            'f1_micro': f1_score(y_hat, y_test, average='micro'),
            'f1_macro': f1_score(y_hat, y_test, average='macro'),
            # 'model_file': f'acc{score_test}_seed{random_state}.sav',
            'params': model.best_params_,
            'cv_results': model.cv_results_
        }

        JSON.create_json_file(
            f'tests/perceptron/{library}/info.json', dump_info, cls=NumpyEncoder)

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

        # param_grid = {
        #     'C': [0.1, 1, 10, 100, 1000],
        #     'kernel': ['linear', 'rbf', 'poly'],
        #     'decision_function_shape': ['ovo', 'ovr']
        # }

        param_grid = {
            'C': [10],
            'kernel': ['linear'],
            'decision_function_shape': ['ovo']
        }

        model = GridSearchCV(svm.SVC(), param_grid, cv=5,
                             refit=True, verbose=2, n_jobs=-1)

        model.fit(X_train, y_train)

        best_params = model.best_params_

        score_test = model.score(X_test, y_test)

        score_train = model.score(X_train, y_train)

        y_hat = model.predict(X_test)

        # SALVA ACURÁCIAS E PARAMETROS
        dump_info = {
            'method': 'Grid Search Support Vector Machines',
            'seed': random_state,
            'library': library,
            'feature_method': method_algo,
            'sample_rate': sampling_rate,
            'train_test': [len(X_train), len(X_test)],
            'score_train': score_train,
            'score_test': score_test,
            'f1_micro': f1_score(y_hat, y_test, average='micro'),
            'f1_macro': f1_score(y_hat, y_test, average='macro'),
            # 'model_file': f'acc{f1_score(y_hat, y_test, average="macro")}_seed{random_state}.sav',
            'params': model.best_params_,
            'cv_results': model.cv_results_
        }

        JSON.create_json_file(
            f'tests/svm/{library}/info.json', dump_info, cls=NumpyEncoder)
