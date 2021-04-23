from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, librosa.display

num_cores = multiprocessing.cpu_count()

f = {}

mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = './archive/VCTK-Corpus/VCTK-Corpus/merged_teste'

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break


def process_directory(dir, index):
    signal, sr = [], 22050

    max = -10000

    signals = []

    for j, audioname in enumerate(f[dir]):
        if j < 1:
            holder_signal, _ = librosa.load(f'{mypath}/{dir}/{audioname}', sr=sr)

            intervals = librosa.effects.split(holder_signal, top_db=20)

            audio_temp = holder_signal[intervals[0][0]:intervals[-1][-1]]

            signal.extend(audio_temp)

            if max < audio_temp.shape[0]:
                max = audio_temp.shape[0]

    for j, audioname in enumerate(f[dir]):
        if j < 1:
            holder_signal, _ = librosa.load(f'{mypath}/{dir}/{audioname}', sr=sr)

            intervals = librosa.effects.split(holder_signal, top_db=20)

            holder_signal = holder_signal[intervals[0][0]:intervals[-1][-1]]

            audio_temp = np.zeros((max,))

            audio_temp[:holder_signal.shape[0]] = holder_signal

            signals.append(audio_temp)

    signal = np.array(signal)
    signals = np.array(signals)

    n_fft = 2048
    hop_length = 16

    mfcc_all = librosa.feature.mfcc(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    mfcc_signal = []

    for i in range(signals.shape[0]):
        mfcc = librosa.feature.mfcc(signals[i], sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
        mfcc_signal.append(mfcc)

    mfcc_signal = np.array(mfcc_signal)

    media_mfcc_signal = mfcc_signal.mean(0)

    librosa.display.specshow(mfcc_all, sr=sr, hop_length=hop_length)
    plt.title('MFCC Merge')
    plt.show()

    librosa.display.specshow(media_mfcc_signal, sr=sr, hop_length=hop_length)
    plt.title('MFCC Média')
    plt.show()


if __name__ == '__main__':
    for j, i in enumerate(list(f.keys())):
        if j < 10:
            process_directory(i, j)
            input('executar proximo!')
        else:
            break
    # signal, samplerate = librosa.load('./archive/VCTK-Corpus/VCTK-Corpus/wav48/p227/p227_002.wav')
    #
    # intervals = librosa.effects.split(signal, top_db=20)
    #
    # audio_temp = signal[intervals[0][0]:intervals[-1][-1]]
    #
    # sf.write('audio_temp2.wav', audio_temp, samplerate=samplerate)
    #
    # print(signal.shape)
    # print(intervals)

    # m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
    #     delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 1)
