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


# %%
# %% Load dataset
sampling_rate = 24000

method_algo = 'mfcc'

library = 'psf'


x, y, mapping = Directory.load_json_data(f'processed/{method_algo}/{library}/{method_algo}_{sampling_rate}.json',
                                         inputs_fieldname=method_algo)


# %%
random_state = 42
x_holder = []

for row in x:
    x_holder.append(row.flatten())

x = np.array(x_holder)

n = len(x)
# n = 2000

x = x[:n]
y = y[:n]

x_train, x_test, y_train, y_test = train_test_split(x, y)


# %%
# Number of trees in random forest
n_estimators = list(range(200, 601, 200))
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [4, 5, 6, 7, 8]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split': min_samples_split,
              'min_samples_leaf': min_samples_leaf,
              'bootstrap': bootstrap}


# %%
model = GridSearchCV(RandomForestClassifier(), param_grid,
                     cv=3, refit=True, verbose=2, n_jobs=-1)

model.fit(x_train, y_train)


# %%
# # print best parameter after tuning
best_params = model.best_params_
print(best_params)

# TESTA ACCURÁCIAS

score_test = model.score(x_test, y_test)

score_train = model.score(x_train, y_train)

y_hat = model.predict(x_test)

# SALVA MODELO
filename = f'models/randomforest/{method_algo}_{sampling_rate}_{best_params["n_estimators"]}_{best_params["max_depth"]}/acc{score_test}_seed{random_state}.sav'

Directory.create_directory(filename, file=True)

joblib.dump(model, filename)

# SALVA ACURÁCIAS E PARAMETROS
dump_info = {
    'method': 'Grid Search Random Forest',
    'seed': random_state,
    'feature_method': method_algo,
    'sample_rate': sampling_rate,
    'train_test': [len(x_train), len(x_test)],
    'score_train': score_train,
    'score_test': score_test,
    'f1_micro': f1_score(y_hat, y_test, average='micro'),
    'f1_macro': f1_score(y_hat, y_test, average='macro'),
    'model_file': f'acc{score_test}_seed{random_state}.sav',
    'params': model.best_params_,
    'cv_results': model.cv_results_
}


# %%
JSON.create_json_file(
    f'models/randomforest/{method_algo}_{sampling_rate}_{best_params["n_estimators"]}_{best_params["max_depth"]}/info.json', dump_info)


# %%
