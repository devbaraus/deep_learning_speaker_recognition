# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:58:01 2021

@author: hugov
"""

import librosa
import python_speech_features
import matplotlib.pyplot as plt
from scipy.signal.windows import hann
import seaborn as sns
from deep_audio import Visualization, Audio

n_mfcc = 13
n_mels = 40
n_fft = 512 
hop_length = 160
fmin = 0
fmax = None
sr = 16000
y, sr = librosa.load(librosa.util.example_audio_file(), sr=sr, duration=5,offset=30)

mfcc_librosa = librosa.feature.mfcc(y=y, sr=sr, n_fft=n_fft,
                                    n_mfcc=n_mfcc, n_mels=n_mels,
                                    hop_length=hop_length,
                                    fmin=fmin, fmax=fmax, htk=False)
Visualization.plot_cepstrals(mfcc_librosa, show=True, cmap='viridis')


mfcc_speech = python_speech_features.mfcc(signal=y, samplerate=sr, winlen=n_fft / sr, winstep=hop_length / sr,
                                          numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                                          preemph=0.0, ceplifter=0, appendEnergy=False, winfunc=hann)
Visualization.plot_cepstrals(mfcc_speech.T, show=True, cmap='viridis')