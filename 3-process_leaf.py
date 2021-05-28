#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed
import numpy as np
import multiprocessing
from tensorflow import newaxis
import leaf_audio.frontend as frontend
from deep_audio import Directory, Audio, Process

num_cores = multiprocessing.cpu_count()
sampling_rate = 24000
path = f'audios/{sampling_rate}'

f = Directory.filenames(path)

# quantidade de segmentos
n_segments = 10
# quantidade de audios
n_audios = 40


def process_directory(dir, index, library):
    signal, rate = Audio.read(
        f'{path}/{dir}', sr=sampling_rate, normalize=True)

    signal = np.array(signal)

    # arredonda o sinal de audio para multiplo de 5
    signal = signal[:len(signal) - len(signal) % (rate * 5)]

    # avalia quantos segmentos têm em uma audio
    segments = len(signal) // (rate * 5)

    m = {
        'attrs': [],
        'labels': [index] * segments
    }

    for i in range(segments):
        if n_segments and i >= n_segments:
            continue

        sample = Audio.segment(signal, rate, seconds=5, window=i)
        sample = sample[newaxis, :]

        if library == 'leaf':
            leaf = frontend.Leaf()
            attr = leaf(sample)
        elif library == 'melbanks':
            melfbanks = frontend.MelFilterbanks()
            attr = melfbanks(sample)
        elif library == 'tfbanks':
            tfbanks = frontend.TimeDomainFilterbanks()
            attr = tfbanks(sample)
        elif library == 'sincnet':
            sincnet = frontend.SincNet()
            attr = sincnet(sample)
        elif library == 'sincnetplus':
            sincnet_plus = frontend.SincNetPlus()
            attr = sincnet_plus(sample)

        attr = np.array(attr)

        m['attrs'].append(attr.tolist())

        del attr
    del signal
    return m


if __name__ == '__main__':
    #     for library in ['leaf', 'melbanks', 'tfbanks', 'sincnet', 'sincnetplus']:
    #         m = []
    #         for j, i in enumerate(f):
    #             if j  < 5:
    #                 m.append(process_directory(i, j, library))

    #         object_mfcc_to_json(m, library)

    for library in ['melbanks', 'tfbanks', 'sincnet', 'sincnetplus']:
        m = Parallel(n_jobs=num_cores, verbose=len(f))(
            delayed(process_directory)
            (i, j, library)
            for j, i in enumerate(f)
            if n_audios and j < n_audios
        )
        Process.object_to_json(f'processed/leaf/{library}_{sampling_rate}.json', m, f)
        del m
