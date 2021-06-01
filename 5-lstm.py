# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dropout, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import time
import matplotlib.pyplot as plt
from numpy import squeeze
from deep_audio import Directory, JSON


# %%
method_algo = 'melbanks'
n_audios = 109
n_segments = 50
model_algo = 'lstm'
n_rate = 24000

DATASET_PATH = f'processed/{method_algo}_{n_audios}-{n_segments}_{n_rate}.json'

# DATASET_PATH = f'processed/mfcc/librosa/mfcc_{n_rate}.json'

X, y, mapping = Directory.load_json_data(DATASET_PATH)

X = squeeze(X, axis=3)

# %%
# SPLIT DOS DADOS
random_state = 42
# for random_state in [5438, 53, 14]:
#     for _ in range(4):
# split data into train and test set
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

# %%


def create_dense_model():
    # build the network architecture
    model = Sequential([
        # 1st hidden layer
        LSTM(512, input_shape=[X.shape[1],
             X.shape[2]], return_sequences=True),
        LSTM(256),
        Dense(128, activation='relu'),
        Dropout(0.3),
        # output layer
        Dense(len(mapping), activation='softmax'),
    ])

    optimizer = Adam(learning_rate=0.0001)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


# %%
# CRIA O MODELO
print(X_train.shape)
model = create_dense_model()


# %%
# SALVA A ESTRUTURA DO MODELO

timestamp = int(time.time())

Directory.create_directory(f'models/{model_algo}/{method_algo}/{timestamp}')

JSON.create_json_file(
    f'models/{model_algo}/{method_algo}/{timestamp}/model_structure.json', model.to_json())

model_save_filename = f'models/{model_algo}/{method_algo}/{timestamp}/model_weight.h5'

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
    f'models/{model_algo}/{method_algo}/{timestamp}/graph_accuracy.png')
plt.close()

# GERA O GRÁFICO DE PERCA
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.savefig(f'models/{model_algo}/{method_algo}/{timestamp}/graph_loss.png')
plt.close()


# %%
# PEGA A MAIOR E ACURÁCIA
higher_accuracy = model.evaluate(X_test, y_test, batch_size=128)

higher_accuracy = str(int(higher_accuracy[1] * 10000)).zfill(4)


# %%
# RENOMEIA A PASTA
Directory.rename_directory(f'models/{model_algo}/{method_algo}/{timestamp}',
                           f'models/{model_algo}/{method_algo}/acc{higher_accuracy}_seed{random_state}_epochs{len(history.history["accuracy"])}_time{timestamp}')


# %%
