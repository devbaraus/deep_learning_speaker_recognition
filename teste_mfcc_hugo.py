import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from spafe.utils import vis
from spafe.features.lpc import lpc, lpcc
from spafe.features.mfcc import mfcc
import numpy as np

path2='archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_006.wav'; flagMono2 = 1

sc_srate2, sc_sample2 = read(path2)
if flagMono2 == 0:
    # sc_sample2 = sc_sample2[:, 0]
    sc_sample2 = np.mean(sc_sample2, axis=1, dtype=type(sc_sample2[0]))

# lb_sample1 = lb_sample1 * 32768
# lb_sample2 = lb_sample2 * 32768
# sc_sample1 = sc_sample1 * 32768
# sc_sample2 = sc_sample2 * 32768


# MFCC
num_ceps_MFCC = 40
nfilts = 512
nfft = 2048

# LPCC
num_ceps_LPCC = 27
lifter = 0
normalize = False

mfccs2 = mfcc(sig=sc_sample2, fs=sc_srate2, num_ceps=num_ceps_MFCC, nfilts=nfilts, nfft=nfft, pre_emph=0)
lpccs2 = lpcc(sig=sc_sample2, fs=sc_srate2, num_ceps=num_ceps_LPCC, lifter=lifter, normalize=normalize, pre_emph=0)

##############################

plt.subplot(3, 1, 1)
plt.title(f'{path2}')
plt.imshow(mfccs2.T,
           origin='lower',
           aspect='auto',
           cmap='magma',
           interpolation='nearest')
plt.xlabel('Frame Index')
plt.ylabel('MFCC Index')


plt.subplot(3, 1, 2)
plt.plot(sc_sample2)

plt.subplot(3, 1, 3)
plt.imshow(lpccs2.T,
           origin='lower',
           aspect='auto',
           cmap='magma',
           interpolation='nearest')
plt.xlabel('Frame Index')
plt.ylabel('LPCC Index')

plt.show(block=False)
plt.close()