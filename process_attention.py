#!/usr/bin/env python
# coding: utf-8
# %%
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
from tensorflow import newaxis
from scipy.signal.windows import hann
from leaf_audio.frontend import MelFilterbanks
from python_speech_features import mfcc
from deep_audio import Directory, Audio, Process

# %%
num_cores = multiprocessing.cpu_count()
# amostra do sinal
sampling_rate = 24000
# quantidade de segmentos
n_segments = 50
# quantidade de audios
n_audios = 75
# bibliotecas
libraries = ['melbanks', 'psf']
# lingua
language = 'english'
# caminho para os audios
path = f'{language}/audios/{sampling_rate}'

f = Directory.filenames(path)


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
        'labels': [index] * (n_segments or segments)
    }

    for i in range(segments):
        if n_segments and i >= n_segments:
            continue

        sample = Audio.segment(signal, rate, seconds=5, window=i)

        n_mfcc = 13
        n_mels = 26
        n_fft = 2048
        # Janela e overlapping (em amostras)
        hop_length = 512
        win_length = 1024
        # Janela e overlapping (em tempo)
        win_len = win_length / rate
        win_hop = hop_length / rate
        lifter = 22
        fmin = 0
        fmax = rate / 2
        coef_pre_enfase = 0.97
        append_energy = 0

        if library == 'melbanks':
            sample = sample[newaxis, :]
            melfbanks = MelFilterbanks(sample_rate=rate)
            attr = melfbanks(sample)
            attr = np.array(attr).T

        if library == 'psf':
            attr = mfcc(
                signal=sample,
                samplerate=rate,
                winlen=win_len,
                winstep=win_hop,
                numcep=n_mfcc,
                nfilt=n_mels,
                nfft=n_fft,
                lowfreq=fmin,
                highfreq=fmax,
                preemph=coef_pre_enfase,
                ceplifter=lifter,
                appendEnergy=append_energy,
                winfunc=hann
            )
            attr = np.array(attr)

        m['attrs'].append(attr.tolist())

        del attr
    del signal
    return m


if __name__ == '__main__':

    # for library in libraries:
    #     m = []
    #     for j, i in enumerate(f):
    #         if j < 5:
    #             m.append(process_directory(i, j, library))

    #     Process.object_to_json(m, library)


    for library in libraries:
        filename = Directory.processed_filename(language, library, sampling_rate, n_audios, n_segments)

        m = Parallel(n_jobs=num_cores, verbose=len(f))(
            delayed(process_directory)
            (i, j, library)
            for j, i in enumerate(f)
            if n_audios and j < n_audios
        )
        Process.object_to_attention(
            filename,
            m,
            f,
        )
        del m
