import librosa
from tensorflow.keras.models import model_from_json
import tensorflow.keras as keras
from utils import load_json_data, audio_mfcc
import numpy as np


def load_model(model_path):
    # load json and create model
    json_file = open(f'{model_path}/model_structure.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights(f'{model_path}/model_weight.h5')
    return loaded_model


model = load_model('models/acc8933_seed14_epochs1015_time1607618363')

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

inputs_test, targets_test, mapping = load_json_data('datatest/datatest_14_14157.json')
inputs_inf, targets_inf, _ = load_json_data('processed/mfcc/mfcc_80inferencia_parallel.json')

person = 'p229'

song_mfccs = []

signal, sr = librosa.load(f'audios/inferencia/{person}.wav', sr=22050)

song_duration = len(signal) // sr

for slide in list(range(0, song_duration, 10)):
    song_mfccs.append(audio_mfcc(f'audios/inferencia/{person}.wav', slide))


print('Total of ', len(song_mfccs), ' to person ', person)

song_mfccs = np.array(song_mfccs)

predictions = model.predict_classes(song_mfccs)

unique, counts = np.unique(predictions, return_counts=True)

for index, value in enumerate(unique.tolist()):
    print(f'%%%% {mapping[value]} : {counts[index]} count')
