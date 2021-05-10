import librosa
import librosa.display
import numpy as np
from deep_audio import Visualization, Audio
import matplotlib.pyplot as plt
import scipy
import spafe.features.mfcc
import python_speech_features
from scipy.signal.windows import hann
import torchaudio.transforms
import torch
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



path2 = 'archive/VCTK-Corpus/VCTK-Corpus/wav48/p226/p226_034.wav'

signal, rate = Audio.read(path2, sr=24000, normalize=True)
Audio.write('./p225_06_24000.wav', signal, rate)

n_mfcc = 13
n_mels = 26
n_fft = 2048
# Janela e overlapping (em amostras)
hop_length = 512
win_length = 1024
# Janela e overlapping (em tempo)
win_len=win_length/rate
win_hop=hop_length/rate
lifter=22
dct_type=2
norm='ortho'
fmin=0
fmax=rate/2
coef_pre_enfase=0.97
Append_Energy=0

#######
mfcc1 = librosa.feature.mfcc(signal, rate, n_mfcc=n_mfcc, n_fft=n_fft, win_length=hop_length, dct_type=2, norm='ortho', window=scipy.signal.windows.hann, hop_length=hop_length, lifter=lifter, fmin=fmin, fmax=fmax)
mfcc1 = mfcc1 - np.min(mfcc1)
mfcc1 = mfcc1 / np.max(mfcc1)
Visualization.plot_cepstrals(mfcc1.T, title='MFCC librosa', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_librosa')

######
mfcc2 = spafe.features.mfcc.mfcc(signal, rate, num_ceps=n_mfcc, nfilts=n_mels, dct_type=2, nfft=n_fft, pre_emph=1, pre_emph_coeff=coef_pre_enfase, win_type='hamming', normalize=1, lifter=lifter, win_len=win_len, win_hop=win_hop, low_freq=fmin, high_freq=fmax, use_energy=Append_Energy)
mfcc2 = mfcc2 - np.min(mfcc2)
mfcc2 = mfcc2 / np.max(mfcc2)
Visualization.plot_cepstrals(mfcc2, title='MFCC Spafe', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_spafe')

####### 
mfcc3 = python_speech_features.mfcc(signal=signal, samplerate=rate, winlen=win_len, winstep=win_hop,
                                    numcep=n_mfcc, nfilt=n_mels, nfft=n_fft, lowfreq=fmin, highfreq=fmax,
                                    preemph=coef_pre_enfase, ceplifter=lifter, appendEnergy=Append_Energy, winfunc=hann)
Visualization.plot_cepstrals(mfcc3, title='MFCC PSF', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_psf')

melkwargs={"n_fft" : n_fft, "n_mels" : n_mels, "hop_length":hop_length, "f_min" : fmin, "f_max" : fmax}


###### Torchaudio 'textbook' log mel scale 
mfcc4 = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=n_mfcc,                                     dct_type=2, norm='ortho', log_mels=True, melkwargs=melkwargs)(torch.from_numpy(signal))

Visualization.plot_cepstrals(mfcc4.T, title='MFCC Torchaudio textbook log mel', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_torchaudio_textbook')

##### Torchaudio 'librosa compatible' default dB mel scale 
mfcc5 = torchaudio.transforms.MFCC(sample_rate=rate, n_mfcc=n_mfcc, dct_type=2, norm='ortho', log_mels=False, melkwargs=melkwargs)(torch.from_numpy(signal))

Visualization.plot_cepstrals(mfcc5.T, title='MFCC Torchaudio \'librosa\'', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_torchaudio_librosa')



####### Tensorflow
# A 1024-point STFT with frames of 20 ms and 50% overlap.
stfts = tf.signal.stft(signal, frame_length=win_length, frame_step=hop_length,                  fft_length=n_fft)
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
mfccTemp = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrograms)[..., :n_mfcc]

mfcc6 = np.array(mfccTemp)

# mfccs6 = np.sum(mfccs6, axis=2)
Visualization.plot_cepstrals(mfcc6, title='MFCC Tensorflow', x_label='Índices de Frame', y_label='MFCC Índices', show=True, cmap='viridis', fig_name='images/apresentacao/mfcc_tensorflow')
