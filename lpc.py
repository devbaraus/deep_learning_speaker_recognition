import librosa

pathsong = 'archive/VCTK-Corpus/VCTK-Corpus/wav48/p225/p225_001.wav'

sample, sr = librosa.load(pathsong, sr=22050)

lpc = librosa.core.lpc(sample, 16)

print(len(lpc), len(sample))