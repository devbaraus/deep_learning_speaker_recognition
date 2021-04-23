# %% Package imports
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from deep_audio import Audio, Visualization, Directory, Model
import numpy as np

# %% Load dataset
x, y, mapping = Directory.load_json_data('processed/mfcc/mfcc_16000.json',
                                         inputs_fieldname='mfcc')

# %% preprocessing
x_holder = []

for row in x:
    x_holder.append(row.flatten())

x = np.array(x_holder)


x, y = shuffle(x, y)

# n = 3000

# x = x[:n]
# y = y[:n]


# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# kf.get_n_splits(x)

x_train, x_val, y_train, y_val = train_test_split(x, y, train_size=0.8, random_state=42, stratify=y)

# for train_index, test_index in kf.split(x):
#     x_train, x_val = x[train_index], x[test_index]
#     y_train, y_val = y[train_index], y[test_index]

# %% Split data

param_grid = {
    'C': [0.1, 1, 10, 100, 1000],
    # 'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
    'kernel': ['linear', 'rbf', 'poly'],
    'decision_function_shape': ['ovo', 'ovr']
}

# %% training

# model = RandomForestClassifier()
# model = svm.SVC(C=10, kernel='rbf')

model = GridSearchCV(svm.SVC(), param_grid, refit=True, verbose=3, n_jobs=8)

model.fit(x_train, y_train)

# %%

# # print best parameter after tuning
print(model.best_params_)
#
# # print how our model looks after hyper-parameter tuning
print(model.best_estimator_)

grid_predictions = model.predict(x_val)
#
# # print classification report
print(classification_report(y_val, grid_predictions))

# %% SCORE

print(model.score(x_val, y_val))

print(model.predict(np.array([x_val[3]])), y_val[3])
