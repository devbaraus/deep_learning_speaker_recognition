#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
import librosa
import python_speech_features
import numpy as np
import multiprocessing
import scipy
import librosa
import python_speech_features
import torchaudio.transforms
import torch
import spafe.features.mfcc
import tensorflow as tf
from deep_audio import Directory, JSON, Audio, NumpyEncoder


# In[2]:


num_cores = multiprocessing.cpu_count()

sampling_rate = 24000

path = f'audios/{sampling_rate}'

f = Directory.filenames(path)


# In[3]:


def process_directory(dir, index, library):
    signal, rate = Audio.read(
        f'{path}/{dir}', sr=sampling_rate, normalize=True)

    signal = np.array(signal)

    signal = signal[:len(signal) - len(signal) % (rate * 5)]

    segments = len(signal) // (rate * 5)

    m = {
        "mfcc": [],
        "labels": [index] * segments
    }

    for i in range(segments):
        start_sample = rate * i * 5
        finish_sample = start_sample + (rate * 5)

        sample = signal[start_sample:finish_sample]

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
        dct_type = 2
        norm = 'ortho'
        fmin = 0
        fmax = rate / 2
        coef_pre_enfase = 0.97
        append_Energy = 0

        if library == 'librosa':
            mfcc = librosa.feature.mfcc(sample, rate, n_mfcc=n_mfcc, n_fft=n_fft, win_length=hop_length, dct_type=2,
                                        norm='ortho', window=scipy.signal.windows.hann, hop_length=hop_length,
                                        lifter=lifter, fmin=fmin, fmax=fmax)

        elif library == 'spafe':
            mfcc = spafe.features.mfcc.mfcc(sample, rate, num_ceps=n_mfcc, nfilts=n_mels, dct_type=2, nfft=n_fft,
                                            pre_emph=1, pre_emph_coeff=coef_pre_enfase,
                                            win_type='hamming', normalize=1, lifter=lifter, win_len=win_len,
                                            win_hop=win_hop, low_freq=fmin, high_freq=fmax, use_energy=append_Energy)

        elif library == 'torchaudio_textbook':
            melkwargs = {"n_fft": n_fft, "n_mels": n_mels,
                         "hop_length": hop_length, "f_min": fmin, "f_max": fmax}

            mfcc = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=n_mfcc,
                                              dct_type=2, norm='ortho', log_mels=True, melkwargs=melkwargs)(
                torch.from_numpy(sample))

        elif library == 'torchaudio_librosa':
            melkwargs = {"n_fft": n_fft, "n_mels": n_mels,
                         "hop_length": hop_length, "f_min": fmin, "f_max": fmax}

            mfcc = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=n_mfcc,
                                              dct_type=2, norm='ortho', log_mels=False, melkwargs=melkwargs)(
                torch.from_numpy(sample))

        elif library == 'psf':
            mfcc = python_speech_features.mfcc(signal=sample, samplerate=rate, winlen=win_len, winstep=win_hop,
                                               numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                                               preemph=coef_pre_enfase, ceplifter=lifter, appendEnergy=append_Energy,
                                               winfunc=scipy.signal.windows.hann)

        elif library == 'tensorflow':
            stfts = tf.signal.stft(sample, frame_length=win_length,
                                   frame_step=hop_length, fft_length=n_fft)
            spectrograms = tf.abs(stfts)

            # Warp the linear scale spectrograms into the mel-scale.
            num_spectrogram_bins = stfts.shape[-1]
            lower_edge_hertz, upper_edge_hertz, num_mel_bins = fmin, fmax, n_mels
            linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                num_mel_bins, num_spectrogram_bins, rate, lower_edge_hertz,
                upper_edge_hertz)
            mel_spectrograms = tf.tensordot(
                spectrograms, linear_to_mel_weight_matrix, 1)
            mel_spectrograms.set_shape(spectrograms.shape[:-1].concatenate(
                linear_to_mel_weight_matrix.shape[-1:]))

            # Compute a stabilized log to get log-magnitude mel-scale spectrograms.
            log_mel_spectrograms = tf.math.log(mel_spectrograms + 1e-6)

            # Compute MFCCs from log_mel_spectrograms and take the first 13.
            mfccTemp = tf.signal.mfccs_from_log_mel_spectrograms(
                log_mel_spectrograms)[..., :n_mfcc]

            mfcc = np.array(mfccTemp)

        m['mfcc'].append(mfcc.tolist())

    print(f'{dir} -> segments: {segments}')
    return m


# In[4]:


def object_mfcc_to_json(m, library):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    data['mapping'] = [file.replace('.wav', '') for i, file in enumerate(f)]

    for i in m:
        data['mfcc'].extend(i['mfcc'])
        data['labels'].extend(i['labels'])

    print('Writing')

    JSON.create_json_file(
        f'processed/mfcc/{library}/mfcc_{sampling_rate}.json', data, cls=NumpyEncoder)

    del data


# In[ ]:


if __name__ == '__main__':
    # for library in ['librosa', 'psf', 'torchaudio', 'tensorflow', 'spafe']:
    #     m = []
    #     for j, i in enumerate(f):
    #         m.append(process_directory(i, j, library))
    #
    #     object_mfcc_to_json(m, library)

    for library in ['torchaudio_librosa', 'torchaudio_textbook', 'librosa', 'psf', 'tensorflow', 'spafe']:
        m = Parallel(n_jobs=num_cores // 2, verbose=len(f))(
            delayed(process_directory)(i, j, library) for j, i in enumerate(f))
        object_mfcc_to_json(m, library)


# In[ ]:


# In[ ]:
