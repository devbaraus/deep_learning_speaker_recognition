import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import spafe
import librosa
import librosa.display
from scipy.io.wavfile import read
from spafe.features.lpc import lpc, lpcc
from spafe.utils import vis

sc_srate, sc_sample = read('archive/hoogle/xd/normal.wav')
lb_sample, lb_srate = librosa.load('archive/hoogle/xd/normal.wav', mono=True, sr=None)
# sc_sample = sc_sample[:, 0]
print(np.max(sc_sample), np.max(lb_sample), np.min(lb_sample))

# # trim scipy
# intervals = librosa.effects.split(sc_sample.astype(np.float32), top_db=20)
#
# sc_trim = sc_sample[intervals[0][0]:intervals[-1][-1]]
#
# # trim scipy
# intervals = librosa.effects.split(lb_sample.astype(np.float32), top_db=20)
#
# lb_trim = lb_sample[intervals[0][0]:intervals[-1][-1]]
#
# print(lb_trim[24], sc_trim[24])
