import librosa
import matplotlib.pyplot as plt
import numpy as np

person = "p225"
person2 = "p225"

pathsong = f'./archive/VCTK-Corpus/VCTK-Corpus/wav48/{person}/{person}_001.wav'
pathsong2 = f'./archive/VCTK-Corpus/VCTK-Corpus/wav48/{person2}/{person2}_001.wav'

sample, sr = librosa.load(pathsong, sr=22050)
sample2, sr2 = librosa.load(pathsong2, sr=22050)

length = 1024

lpc = librosa.core.lpc(sample, 16)
lpc2 = librosa.core.lpc(sample2, 32)


lpc = np.abs(lpc)
lpc2 = np.abs(lpc2)

plt.plot(lpc)
plt.plot(lpc2)
plt.legend(['ordem 1024', 'ordem 4096'])
plt.xlabel('')
plt.ylabel('LPC')

# plt.plot(sample[::32])
plt.savefig('teste.png')
plt.show()

