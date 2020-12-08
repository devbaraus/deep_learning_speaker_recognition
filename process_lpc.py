from os import walk
from joblib import Parallel, delayed
import librosa
import numpy as np
import multiprocessing
import json
from utils import get_filenames, process_lpc

num_cores = multiprocessing.cpu_count() // 2

path = './archive/VCTK-Corpus/VCTK-Corpus/merged'

f = get_filenames(path)

data = {
    "mapping": [],
    "lpc": [],
    "labels": []
}


def process_directory(dir, index):
    signal, sr = [], 22050

    signal, _ = librosa.load(f'{path}/{dir}', sr=sr)

    signal = np.array(signal)

    signal = signal[:len(signal) - len(signal) % (sr * 5)]

    segments = len(signal) // (sr * 5)

    m = {
        "lpc": [],
        "labels": [index] * segments
    }

    for i in range(segments):
        start_sample = sr * i * 5
        finish_sample = start_sample + (sr * 5)

        lpc = process_lpc(signal[start_sample:finish_sample])

        m['lpc'].append(lpc.tolist())

    print(f'{dir} -> segments: {segments}')
    return m


def object_mfcc_to_json(m):
    data['mapping'] = [file.replace('.wav', '') for file in f]

    for i in m:
        data['lpc'].extend(i['lpc'])
        data['labels'].extend(i['labels'])

    with open('processed/lpc/lpc_abs_parallel.json', 'w') as fp:
        json.dump(data, fp, indent=2)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores, verbose=len(f), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(f) if i is not None)
    object_mfcc_to_json(m)
