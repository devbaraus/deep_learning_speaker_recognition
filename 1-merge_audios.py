#!/usr/bin/env python
# coding: utf-8

from joblib import Parallel, delayed
import multiprocessing
from numpy import array
from deep_audio import Audio, Directory

num_cores = multiprocessing.cpu_count()
mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = f'audios'

n_audios = 40

f = Directory.filenames_recursive(mypath)


def process_directory(dir, n_rate):
    signal = []

    for j, audioname in enumerate(f[dir]):
        holder_signal, sr = Audio.read(
            f'{mypath}/{dir}/{audioname}', sr=n_rate)

        signal.extend(Audio.trim(holder_signal, 20))

    signal = array(signal)

    Audio.write(f'{destpath}/{n_rate}/{n_audios}/{dir}.wav', signal, n_rate)


if __name__ == '__main__':
    # for j, i in enumerate(list(f.keys())):
    #     if n_audios and j < n_audios:
    #         for rate in [24000]:
    #             process_directory(i, rate)
    Parallel(n_jobs=num_cores // 2, verbose=len(f.keys()))(
        delayed(process_directory)(i, rate)
        for j, i in enumerate(list(f.keys()))
        if n_audios and j < n_audios
        for rate in [24000]
    )
