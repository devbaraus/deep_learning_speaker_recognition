#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %% Package imports
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from deep_audio import Audio, Visualization, Directory, Model, JSON
import numpy as np
from joblib import Parallel, delayed
import multiprocessing
import joblib


# In[7]:


# %% Load dataset
sampling_rate = 24000

method_algo = 'mfcc'
library = 'librosa'

x, y, mapping = Directory.load_json_data(f'processed/{method_algo}/{library}/{method_algo}_{sampling_rate}.json',
                                         inputs_fieldname=method_algo)


# In[8]:


random_state = 42
x_holder = []

for row in x:
    x_holder.append(row.flatten())

x = np.array(x_holder)

n = len(x)
# n = 2000

x = x[:n]
y = y[:n]

x_train, x_test, y_train, y_test = train_test_split(x,
                                                      y,
                                                      test_size=0.2,
                                                      stratify=y,
                                                      random_state=random_state)

x_train, x_valid, y_train, y_valid = train_test_split(x_train,
                                                    y_train,
                                                    test_size=0.2,
                                                    stratify=y_train,
                                                    random_state=random_state)


# In[9]:


param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'poly'],
    'decision_function_shape': ['ovo', 'ovr']
}

param_grid = {
    'C': [10],
    'kernel': ['linear'],
    'decision_function_shape': ['ovo']
}

model = GridSearchCV(svm.SVC(), param_grid, cv=5, refit=True, verbose=2, n_jobs=8)

model.fit(x_train, y_train)


# In[10]:


# # print best parameter after tuning
best_params = model.best_params_
print(best_params)

# TESTA ACCURÁCIAS

score_valid = model.score(x_valid, y_valid)

score_train = model.score(x_train, y_train)

y_hat = model.predict(x_test)

# SALVA MODELO
filename = f'models/gridsvm/{method_algo}_{f1_score(y_hat, y_test, average="macro")}_{sampling_rate}_{best_params["kernel"]}_{best_params["decision_function_shape"]}/acc{f1_score(y_hat, y_test, average="macro")}_seed{random_state}.sav'

Directory.create_directory(filename, file=True)

joblib.dump(model, filename)

# SALVA ACURÁCIAS E PARAMETROS
dump_info = {
    'method': 'Grid Search Support Vector Machines',
    'seed': random_state,
    'library': library,
    'feature_method': method_algo,
    'sample_rate': sampling_rate,
    'train_test': [len(x_train), len(x_test)],
    'score_train': score_train,
    'score_valid': score_valid,
    'f1_micro': f1_score(y_hat, y_test, average='micro'),
    'f1_macro': f1_score(y_hat, y_test, average='macro'),
    'model_file': f'acc{f1_score(y_hat, y_test, average="macro")}_seed{random_state}.sav',
    'params': model.best_params_,
    'cv_results': model.cv_results_
}


# In[11]:


JSON.create_json_file(f'models/gridsvm/{method_algo}_{f1_score(y_hat, y_test, average="macro")}_{sampling_rate}_{best_params["kernel"]}_{best_params["decision_function_shape"]}/info.json', dump_info)


# In[ ]:




