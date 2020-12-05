# https://www.youtube.com/watch?v=Oa_d-zaUti8&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf&index=11
import librosa, librosa.display
import soundfile  as sf
import matplotlib.pyplot as plt
import numpy as np

signal, sr = [], 22050

for i in range(1, 10):
    file = f'./archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_00{i}.wav'
    holder_signal, _ = librosa.load(file, sr=sr)
    signal.extend(holder_signal)

signal = np.array(signal)

signal = signal[:len(signal) - len(signal) % (sr * 5)]

segments = len(signal) // (sr * 5)

sf.write('file_trim_5s.wav', signal, sr)

n_fft = 2048
hop_length = 16

start_sample = sr * 5
finish_sample = start_sample + (sr * 5)

# MFCCs (mel frequency cepstral coefficients)
audio_segment = librosa.feature.mfcc(signal[:sr * 5], sr=sr, n_fft=n_fft, hop_length=hop_length,
                                     n_mfcc=13)
mfcc_segment = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
# mfcc_segment = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

mfcc_segment = mfcc_segment[:, :(sr * 5 // hop_length)]

np.savetxt('audio_seg.csv', audio_segment)
np.savetxt('mfcc_seg.csv', mfcc_segment)

librosa.display.specshow(mfcc_segment, sr=sr, hop_length=hop_length)
librosa.display.specshow(audio_segment, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()
