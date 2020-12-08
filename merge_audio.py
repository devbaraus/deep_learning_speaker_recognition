from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import soundfile as sf
import librosa

num_cores = multiprocessing.cpu_count()

f = {}

mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = './archive/VCTK-Corpus/VCTK-Corpus/merged'

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
        if j < 10:
            holder_signal, _ = librosa.load(f'{mypath}/{dir}/{audioname}', sr=sr)
            signal.extend(holder_signal)

    signal = np.array(signal)

    sf.write(f'{destpath}/{dir}.wav', signal, sr)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores, verbose=len(f.keys()), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 1)
