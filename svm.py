# %%
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from deep_audio import Directory, Process, Terminal, Model
# %%
args = Terminal.get_args()

language = args['language']
library = args['representation']
people = args['people']
segments = args['segments']
sampling_rate = 24000
random_state = 42
# %%
global X_train, X_valid, X_test, y_train, y_valid, y_test

file_path = Directory.processed_filename(
    language, library, sampling_rate, people, segments)
# %%
if language == 'mixed' and library == 'mixed':
    first_folder = Directory.processed_filename(
        'portuguese', 'psf', sampling_rate, None, None)
    second_folder = Directory.processed_filename(
        'portuguese', 'melbanks', sampling_rate, None, None)
    third_folder = Directory.processed_filename(
        'english', 'psf', sampling_rate, people, segments)
    fourth_folder = Directory.processed_filename(
        'english', 'melbanks', sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection(
        first_folder, second_folder, third_folder, fourth_folder,
        validation=True,
        test=True
    )
elif language == 'mixed':
    portuguese_folder = Directory.processed_filename(
        'portuguese', library, sampling_rate, people, segments)
    english_folder = Directory.processed_filename(
        'english', library, sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_language(
        portuguese_folder=portuguese_folder,
        english_folder=english_folder,
        flat=True
    )
elif library == 'mixed':
    first_folder = Directory.processed_filename(
        language, 'psf', sampling_rate, people, segments)
    second_folder = Directory.processed_filename(
        language, 'melbanks', sampling_rate, people, segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_representation(
        first_folder,
        second_folder,
        test=True)
else:
    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.selection(
        file_path, flat=True)


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

filename_ps = Directory.verify_people_segments(
    people=people, segments=segments)

# SALVA ACURÁCIAS E PARAMETROS
Model.dump_grid(
    f'{language}/models/svm/{library}/{filename_ps}{Process.pad_accuracy(score_test)}_{abs(time())}/info.json',
    model=model,
    method='Support Vector Machines',
    sampling_rate=sampling_rate,
    seed=random_state,
    library=library,
    sizes=[len(X_train), len(X_valid), len(X_test)],
    score_train=score_train,
    score_test=score_test,
)
