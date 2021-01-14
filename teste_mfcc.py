import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.io.wavfile import read
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc
import numpy as np

# init input vars
# num_ceps = 13
# lifter = 0
# normalize = True

# read wav
# path='archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_001.wav'
path='archive/hoogle/xd/normal.wav'
sc_srate, sc_sample = read(path)
lb_sample, lb_srate = librosa.load(path, mono=True, sr=None)


# sig = np.mean(sig, axis=1)
#
# # visualize spectogram
# vis.spectogram(sig=lb_sample, fs=sc_srate)
# # compute lpccs
# lpccs = lpcc(sig=sig, fs=fs, num_ceps=num_ceps, lifter=lifter, normalize=normalize)
# # visualize features
# librosa.display.specshow(lpccs, sr=fs)
# plt.show()

print(sc_sample.shape, lb_sample.shape)

# # init input vars
num_ceps = 40
nfilts = 512
nfft = 2048
#
#
# compute features
mfccs = mfcc(sig=sc_sample, fs=lb_srate, num_ceps=num_ceps, nfilts=nfilts, nfft=nfft, pre_emph=0)
#
# vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index')
# vis.visualize_features(mfccs, 'MFCC Index', 'Frame Index', cmap='coolwarm')
librosa.display.specshow(mfccs.T, sr=lb_srate, y_axis='frames', x_axis='frames')
plt.xlabel('Frame Index')
plt.ylabel('MFCC Index')
plt.colorbar()
plt.show()
plt.close()
#
#
mfcc_librosa = librosa.feature.mfcc(lb_sample, sr=lb_srate, n_fft=nfft, hop_length=nfilts, n_mfcc=num_ceps)
# # #
librosa.display.specshow(mfcc_librosa, sr=lb_srate)
plt.xlabel('Frame Index')
plt.ylabel('MFCC Index')
plt.colorbar()
plt.show()
plt.close()
