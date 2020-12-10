from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import soundfile as sf
import librosa

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

    signals = []

    for j, audioname in enumerate(f[dir]):
        if j < 10:
            holder_signal, _ = librosa.load(f'{mypath}/{dir}/{audioname}', sr=sr)
            signal.extend(holder_signal)
            signals.append(holder_signal)

    signal = np.array(signal)
    signals = np.array(signals)

    lpc_all = librosa.core.lpc(signal, 4096)
    lpc_all = np.abs(lpc_all)

    lpc_signal = []

    for i in range(signals.shape[0]):
        lpc_signal.append(librosa.core.lpc(signals[i], 4096))

    lpc_signal = np.array(lpc_signal)

    media_lpc_signal = lpc_signal.mean(0)
    media_lpc_signal = np.abs(media_lpc_signal)

    plt.plot(lpc_all)
    plt.plot(media_lpc_signal)
    plt.show()


if __name__ == '__main__':
    for j, i in enumerate(list(f.keys())):
        if j < 10:
            process_directory(i, j)
            input('executar proximo!')
        else:
            break
    # m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
    #     delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 1)
