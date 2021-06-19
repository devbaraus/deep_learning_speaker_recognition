# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from numpy import squeeze, max
from deep_audio import Directory, JSON, Process

# %%
model_algo = 'perceptron'
language = 'portuguese'
library = 'psf'
n_people = None
n_segments = None
n_rate = 24000

filename_ps = Directory.verify_people_segments(people=n_people, segments=n_segments)

# DATASET_PATH = Directory.processed_filename(language, library, n_rate, n_people, n_segments)

# X, y, mapping = Directory.load_json_data(DATASET_PATH)

# if library != 'psf':
#     X = squeeze(X, axis=3)

# %%
# SPLIT DOS DADOS
random_state = 42

# split data into train and test set
# X_train, X_test, y_train, y_test = train_test_split(X,
#                                                     y,
#                                                     test_size=0.2,
#                                                     stratify=y,
#                                                     random_state=random_state)
#
# X_train, X_valid, y_train, y_valid = train_test_split(X_train,
#                                                       y_train,
#                                                       test_size=0.2,
#                                                       stratify=y_train,
#                                                       random_state=random_state)

# X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_language(
#     'portuguese/processed/psf_24000.json',
#     'english/processed/p75_s50/psf_24000.json', test=True, flat=True)

X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_representation(
    'portuguese/processed/psf_24000.json',
    'portuguese/processed/melbanks_24000.json', test=True, )

mapping = set(y_test)

print(set(y_train) == set(y_test), set(y_train) == set(y_valid), len(mapping), len(set(y_train)), len(set(y_valid)))


if language == 'mixed' and library == 'mixed':
    first_folder = Directory.processed_filename(
        'portuguese', 'psf', n_rate, None, None)
    second_folder = Directory.processed_filename(
        'portuguese', 'melbanks', n_rate, None, None)
    third_folder = Directory.processed_filename(
        'english', 'psf', n_rate, n_people, n_segments)
    fourth_folder = Directory.processed_filename(
        'english', 'melbanks', n_rate, n_people, n_segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection(
        first_folder, second_folder, third_folder, fourth_folder,
        validation=True,
        test=False
    )
elif language == 'mixed':
    print('MIXED LANGUAGE')
    portuguese_folder = Directory.processed_filename(
        'portuguese', library, n_rate, None, None)
    english_folder = Directory.processed_filename(
        'english', library, n_rate, n_people, n_segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_language(
        portuguese_folder=portuguese_folder,
        english_folder=english_folder,
        validation=True
    )
elif library == 'mixed':
    first_folder = Directory.processed_filename(
        language, 'psf', n_rate, n_people, n_segments)
    second_folder = Directory.processed_filename(
        language, 'melbanks', n_rate, n_people, n_segments)

    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.mixed_selection_representation(
        first_folder,
        second_folder,
        validation=True,
        test=False)
else:
    X_train, X_valid, X_test, y_train, y_valid, y_test = Process.selection(
        DATASET_PATH)


mapping = set(y_train)
# %%


def build_model(learning_rate=0.0001):
    # build the network architecture
    model = Sequential([
        # 1st hidden layer
        Flatten(input_shape=[X_train.shape[0], X_train.shape[1]]),
        Dense(512, activation='relu'),
        Dense(256, activation='relu'),
        Dense(128, activation='relu'),
        Dense(len(mapping) + 2, activation='softmax'),
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# %%
# CRIA O MODELO
learning_rate = 0.0001
model = build_model(learning_rate=learning_rate)

# %%
# SALVA A ESTRUTURA DO MODELO

timestamp = int(time.time())

Directory.create_directory(f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}')

JSON.create_json_file(
    f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}/model_structure.json', model.to_json())

model_save_filename = f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}/model_weight.h5'

# DECIDE QUANDO PARAR
earlystopping_cb = EarlyStopping(patience=300, restore_best_weights=True)

# SALVA OS PESOS
mdlcheckpoint_cb = ModelCheckpoint(
    model_save_filename, monitor="val_accuracy", save_best_only=True
)

# %%
# TREINA O MODELO
history = model.fit(X_train, y_train,
                    validation_data=(X_valid, y_valid),
                    epochs=10000,
                    batch_size=128,
                    callbacks=[earlystopping_cb, mdlcheckpoint_cb])

# %%
# GERA O GRAFICO DE ACURÁCIA
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(
    f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}/graph_accuracy.png')
plt.close()

# GERA O GRÁFICO DE PERCA
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}/graph_loss.png')
plt.close()

# %%
# PEGA A MAIOR E ACURÁCIA
higher_accuracy = model.evaluate(X_test, y_test, batch_size=128)

# %%
dump_info = {
    'learning_rate': learning_rate,
    'language': language,
    'model_algo': model_algo,
    'library': library,
    'people': n_people,
    'segments': n_segments,
    'rate': n_rate,
    'sizes': [X_train.shape, X_valid.shape, X_test.shape],
    'scores': [max(history.history['accuracy']), max(history.history['val_accuracy']), higher_accuracy[1]]
}

JSON.create_json_file(f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}/info.json', dump_info)

# %%
higher_accuracy = str(int(higher_accuracy[1] * 10000)).zfill(4)

# %%
# RENOMEIA A PASTA
Directory.rename_directory(f'{language}/models/{model_algo}/{library}/{filename_ps}{timestamp}',
                           f'{language}/models/{model_algo}/{library}/{filename_ps}acc{higher_accuracy}_seed{random_state}_epochs{len(history.history["accuracy"])}_time{timestamp}')

# %%
