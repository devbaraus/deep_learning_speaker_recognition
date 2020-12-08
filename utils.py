import numpy as np
import json
import librosa
import os


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
