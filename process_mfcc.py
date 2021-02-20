from os import walk
from joblib import Parallel, delayed
import librosa
import numpy as np
import multiprocessing
import json
from deep_audio import Directory, Audio, JSON

num_cores = multiprocessing.cpu_count()

path = 'audios/16000'

f = Directory.filenames(path)

data = {
    "mapping": [],
    "mfcc": [],
    "labels": []
}


def process_directory(dir, index):
    signal, sr = Audio.read(f'{path}/{dir}', normalize=True)

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

        sample = signal[start_sample:finish_sample]

        # sample = Audio.normalize(sample)

        # mfcc = Audio.mfcc(sample, rate=sr)
        mfcc = librosa.feature.mfcc(sample, sr=sr, n_mfcc=13, hop_length=512, n_fft=2048, lifter=22)

        # mfcc = mfcc.T

        m['mfcc'].append(mfcc.tolist())

    print(f'{dir} -> segments: {segments}')
    return m


def object_mfcc_to_json(m):
    data['mapping'] = [file.replace('.wav', '') for file in f]

    for i in m:
        data['mfcc'].extend(i['mfcc'])
        data['labels'].extend(i['labels'])

    JSON.create_json_file('processed/mfcc/mfcc_16000.json', data)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores // 2, verbose=len(f), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(f) if i is not None)
    object_mfcc_to_json(m)
