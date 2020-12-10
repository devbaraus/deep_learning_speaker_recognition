from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import soundfile as sf
import librosa

num_cores = multiprocessing.cpu_count()

f = {}

mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = 'audios'

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break


def process_directory(dir, index):
    signal, sr = [], 22050

    for j, audioname in enumerate(f[dir]):
        # if j < 10:
        holder_signal, _ = librosa.load(f'{mypath}/{dir}/{audioname}', sr=sr)

        intervals = librosa.effects.split(holder_signal, top_db=20)

        audio_temp = holder_signal[intervals[0][0]:intervals[-1][-1]]

        signal.extend(audio_temp)

    treino_size = int(len(signal) * 0.8)

    signal = np.array(signal)

    signal_treino = signal[:treino_size]

    signal_inferencia = signal[treino_size:]

    sf.write(f'{destpath}/treino/{dir}.wav', signal_treino, sr)
    sf.write(f'{destpath}/inferencia/{dir}.wav', signal_inferencia, sr)


if __name__ == '__main__':
    # for j, i in enumerate(list(f.keys())):
    #     if j < 1:
    #         process_directory(i, j)
    m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
        delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())))
