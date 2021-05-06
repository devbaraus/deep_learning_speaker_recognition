# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 17:10:41 2021

@author: hugov
"""

import numpy as np
import librosa
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import torchaudio.transforms
import torch

n_mfcc = 13
n_mels = 40
n_fft = 512 
hop_length = 160
fmin = 0
fmax = None
sr = 16000

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length":hop_length, "f_min" : fmin, "f_max" : fmax}

y, sr = librosa.load(librosa.util.example_audio_file(), sr=sr, duration=5,offset=30)

# Default librosa with db mel scale 
mfcc_lib_db = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)

# Nearly identical to above
# mfcc_lib_db = librosa.feature.mfcc(S=librosa.power_to_db(S), n_mfcc=n_mfcc, htk=False)

# Modified librosa with log mel scale (helper)
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmin=fmin, 
                                    fmax=fmax, hop_length=hop_length)

# Modified librosa with log mel scale
mfcc_lib_log = librosa.feature.mfcc(S=np.log(S+1e-6), n_mfcc=n_mfcc, htk=False)

# Python_speech_features 
mfcc_speech = python_speech_features.mfcc(signal=y, samplerate=sr, winlen=n_fft / sr, winstep=hop_length / sr,
                                          numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                                          preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=hann)

# Torchaudio 'textbook' log mel scale 
mfcc_torch_log = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, 
                                            dct_type=2, norm='ortho', log_mels=True, 
                                            melkwargs=melkwargs)(torch.from_numpy(y))

# Torchaudio 'librosa compatible' default dB mel scale 
mfcc_torch_db = torchaudio.transforms.MFCC(sample_rate=sr, n_mfcc=n_mfcc, 
                                           dct_type=2, norm='ortho', log_mels=False, 
                                           melkwargs=melkwargs)(torch.from_numpy(y))

feature = 3 # <-------- Play with this!!
plt.subplot(2, 1, 1)

plt.plot(mfcc_lib_log.T[:,feature], 'k')
plt.plot(mfcc_lib_db.T[:,feature], 'b')
plt.plot(mfcc_speech[:,feature], 'r')
plt.plot(mfcc_torch_log.T[:,feature], 'c')
plt.plot(mfcc_torch_db.T[:,feature], 'g')
plt.grid()

plt.subplot(2, 2, 3)
plt.plot(mfcc_lib_log.T[:,feature], 'k')
plt.plot(mfcc_torch_log.T[:,feature], 'c')
plt.plot(mfcc_speech[:,feature], 'r')
plt.grid()

plt.subplot(2, 2, 4)
plt.plot(mfcc_lib_db.T[:,feature], 'b')
plt.plot(mfcc_torch_db.T[:,feature], 'g')
plt.grid()