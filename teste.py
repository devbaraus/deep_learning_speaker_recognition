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
fs, sig = read("archive/hoogle/xd/normal.wav")
sample, sr = librosa.load("archive/hoogle/xd/normal.wav", sr=48000)

print('scipy', sig.shape, fs, sig[47][0])
print('librosa', sample.shape, sr, sample[47])

# sig = np.mean(sig, axis=1)
#
# # visualize spectogram
# # vis.spectogram(sig, fs)
# # compute lpccs
# lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# # visualize features
# librosa.display.specshow(lpccs, sr=fs)
# plt.show()

#
# # init input vars
# num_ceps = 13
# nfilts = 512
# nfft = 2058
#
#
# # compute features
# mfccs = mfcc(sig=sig,
#              fs=fs,
#              num_ceps=num_ceps,
#              nfilts=nfilts,
#              nfft=nfft)
#
# vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index')
# vis.visualize_features(np.asarray(mfccs).T, 'MFCC Index', 'Frame Index')
#
#
# mfcc_librosa = librosa.feature.mfcc(sig, sr=fs, n_fft=nfft, hop_length=nfilts, n_mfcc=num_ceps)
#
# vis.visualize_features(mfcc_librosa, 'MFCC Index', 'Frame Index')
# librosa.display.specshow(np.asarray(mfcc_librosa).T, sr=fs)
# plt.xlabel('Frame Index')
# plt.ylabel('MFCC Index')
# plt.show()
# plt.close()
