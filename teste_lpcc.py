import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.io.wavfile import read
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc
import numpy as np

# init input vars
num_ceps = 13
lifter = 0
normalize = True

# read wav
path = 'archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_001.wav'
sc_srate, sc_sample = read(path)
lb_sample, lb_srate = librosa.load(path, mono=True, sr=None)

# # visualize spectogram
# vis.spectogram(sig=sc_sample, fs=sc_srate)
# compute lpccs
lpccs = lpcc(sig=sc_sample, fs=sc_srate, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# visualize features
# vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
librosa.display.specshow(lpccs.T, sr=sc_srate)
plt.show()


lpccs = lpcc(sig=lb_sample, fs=sc_srate, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# visualize features
# vis.visualize_features(lpccs, 'LPCC Index', 'Frame Index')
librosa.display.specshow(lpccs.T, sr=sc_srate)
plt.show()