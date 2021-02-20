# %% Import dos pacotes

import librosa
import numpy as np
from deep_audio import Visualization, Audio

# %% Leitura do áudio

path2 = 'archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_006.wav'

signal, rate = Audio.read(path2, normalize=False)
signal2, _ = Audio.read(path2)

# %% MFCC
num_ceps_MFCC = 13
nfilts = 512
nfft = 2048

mfcc = Audio.mfcc(signal, rate, lifter=22, normalize=0).T
# librosa.feature.melspectrogram
librosa.filters.mel
mfcc1 = librosa.feature.mfcc(signal2, rate, n_mfcc=num_ceps_MFCC, n_fft=nfft, lifter=22)


# %% Visualização
Visualization.plot_cepstrals(mfcc, show=True, close=True)
Visualization.plot_cepstrals(mfcc1.T, show=True, close=True)
