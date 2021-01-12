from os import walk
from joblib import Parallel, delayed
import librosa
import numpy as np
import multiprocessing
import json
from utils import get_filenames, process_mfcc, create_directory

num_cores = multiprocessing.cpu_count()

path = 'audios/inferencia'

f = get_filenames(path)

data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}


def process_directory(dir, index):
    signal, sr = [], 22050

    signal, _ = librosa.load(f'{path}/{dir}', sr=sr)

    signal = np.array(signal)

    signal = signal[:len(signal) - len(signal) % (sr * 5)]

    segments = len(signal) // (sr * 5)

    m = {
        "mfcc": [],
        "labels": [index] * segments
    }

    for i in range(segments):
        start_sample = sr * i * 5
        finish_sample = start_sample + (sr * 5)

        mfcc = process_mfcc(signal[start_sample:finish_sample])

        m['mfcc'].append(mfcc.tolist())

    print(f'{dir} -> segments: {segments}')
    return m


def object_mfcc_to_json(m):
    data['mapping'] = [file.replace('.wav', '') for file in f]

    for i in m:
        data['mfcc'].extend(i['mfcc'])
        data['labels'].extend(i['labels'])

    create_directory('processed/mfcc')

    with open('processed/mfcc/mfcc_80inferencia_parallel.json', 'w') as fp:
        json.dump(data, fp, indent=2)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores, verbose=len(f), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(f) if i is not None)
    object_mfcc_to_json(m)