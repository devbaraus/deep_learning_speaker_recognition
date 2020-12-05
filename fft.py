from scipy.io.wavfile import read
from scipy.signal.windows import hann
from scipy.fftpack import rfft
import numpy as np
import matplotlib.pyplot as plt


samplerate, audio = read('./archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_001.wav')

# plt.plot(audio[:, 0], label='Left Channel')
# plt.xlabel("Amplitude")
# plt.ylabel("Time")
# plt.show()

# apply a Hanning window
window = hann(1024)
audio = audio[0:1024]
# print(audio, '------', np.random.rand(1024, 1))
# audiomod = audio + np.random.rand(1, 1024)[0]
audio2 = audio[0:1024] * window
# fft
mags = abs(rfft(audio))
# mags2 = abs(rfft(audiomod))
mags2 = abs(rfft(audio2))
# convert to dB
# mags = 20 * np.log10(mags)
mags2 = 20 * np.log10(mags2)
# normalise to 0 dB max
# mags -= max(mags)
mags2 -= max(mags2)
plt.plot(mags)
plt.plot(mags2)
# plt.plot(window)
# plt.plot(mags2)
# label the axes
plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency Bin")
# set the title
plt.title("Original")
plt.show()
