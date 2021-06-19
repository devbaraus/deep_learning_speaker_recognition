#!/usr/bin/env python
# coding: utf-8
# %%
from joblib import Parallel, delayed
import multiprocessing
from numpy import array
from deep_audio import Audio, Directory, Terminal
import sys

#%%
args = Terminal.get_args(sys.argv[1:])

# %%
num_cores = multiprocessing.cpu_count()
language = args['language'] or 'portguese'
origin_path = f'base_{language}'
dest_path = f'{language}/audios'
s_rate = [24000]
n_audios = args['people'] or None

print(dest_path)
# %%

f = Directory.filenames_recursive(origin_path)


def process_directory(dir, n_rate):
    signal = []

    for j, audioname in enumerate(f[dir]):
        holder_signal, sr = Audio.read(
            f'{origin_path}/{dir}/{audioname}', sr=n_rate)

        signal.extend(Audio.trim(holder_signal, 20))

    signal = array(signal)

    Audio.write(f'{dest_path}/{n_rate}/{dir}.wav', signal, n_rate)


if __name__ == '__main__':
    # for j, i in enumerate(list(f.keys())):
    #     if n_audios and j < n_audios:
    #         for rate in [24000]:
    #             process_directory(i, rate)
    Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
        delayed(process_directory)(i, rate)
        for j, i in enumerate(list(f.keys()))
        # if n_audios and j < n_audios
        for rate in s_rate
    )
