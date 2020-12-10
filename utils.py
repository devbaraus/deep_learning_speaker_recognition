import numpy as np
import json
import librosa
import os


def create_directory(directory):
    try:
        os.makedirs(directory)
    except FileExistsError:
        pass


def rename_directory(current, newname):
    os.rename(current, newname)


def create_model_json_file(file, data):
    directory = '/'.join(file.split('/')[:-1])

    create_directory(directory)

    with open(file, "w") as json_file:
        json_file.write(data)


def create_json_file(file, data, indent=2):
    directory = '/'.join(file.split('/')[:-1])

    create_directory(directory)

    with open(file, "w") as fp:
        json.dump(data, fp, indent=indent)


def load_json_data(dataset_path, inputs_fieldname='mfcc'):
    with open(dataset_path, 'r') as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data[inputs_fieldname])
    targets = np.array(data['labels'])
    mapping = data['mapping']

    return inputs, targets, mapping


def process_mfcc(signal, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512):
    mfcc = librosa.feature.mfcc(signal,
                                sr=sr,
                                n_mfcc=n_mfcc,
                                n_fft=n_fft,
                                hop_length=hop_length)

    return mfcc


def audio_mfcc(path, slide=0, seconds=5):
    signal, sr = librosa.load(path, sr=22050)
    slide_window = slide * sr
    signal = signal[slide_window: slide_window + seconds * sr]
    mfcc = process_mfcc(signal)
    return mfcc


def process_lpc(signal, length=4095):
    lpc = librosa.core.lpc(signal, length)
    lpc = np.abs(lpc)
    return lpc


def get_filenames(path):
    f = []

    for (_, _, filenames) in os.walk(path):
        f.extend(filenames)
        break

    return f
