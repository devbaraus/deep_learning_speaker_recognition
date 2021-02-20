from os import walk
from joblib import Parallel, delayed
import librosa
import numpy as np
import multiprocessing
import json
from deep_audio import Directory, Audio, JSON

num_cores = multiprocessing.cpu_count()

path = 'audios'

f = Directory.filenames(path)

data = {
    "mapping": [],
    "data": [],
    "labels": []
}


def process_directory(dir, index):
    signal, sr = Audio.read(f'{path}/{dir}')

    signal = np.array(signal)

    signal = signal[:len(signal) - len(signal) % (sr * 5)]

    segments = len(signal) // (sr * 5)

    m = {
        "data": [],
        "labels": [index] * segments
    }

    for i in range(segments):
        start_sample = sr * i * 5
        finish_sample = start_sample + (sr * 5)

        mfcc = Audio.mfcc(signal[start_sample:finish_sample])
        lpcc = Audio.lpcc(signal[start_sample:finish_sample])

        data = np.concatenate((mfcc, lpcc), axis=0)

        m['data'].append(data.tolist())

    print(f'{dir} -> segments: {segments}')
    return m


def object_data_to_json(m):
    data['mapping'] = [file.replace('.wav', '') for file in f]

    for i in m:
        data['data'].extend(i['data'])
        data['labels'].extend(i['labels'])

    JSON.create_json_file('processed/combined/data_80.json', data)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores, verbose=len(f), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(f) if i is not None)
    object_data_to_json(m)
