import multiprocessing
from joblib import Parallel, delayed
from os import walk
from scipy.io.wavfile import read
from scipy.fftpack import rfft
import numpy as np
import time

num_cores = multiprocessing.cpu_count()
f = {}
mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break


def process_fft(path):
    samplerate, audio = read(path)
    audio = audio[0:1024]
    return abs(rfft(audio))


def process_directory(dir):
    processed_audio_matrix = np.zeros((len(f[dir]), 1024))

    for j, audioname in enumerate(f[dir]):
        processed_audio_matrix[j] = process_fft(f'{mypath}/{dir}/{audioname}')

    np.savetxt(f'./processed/{dir}.csv', processed_audio_matrix, delimiter=',', fmt='%10.7f')
    print(f'finished job {dir}')
    return


if __name__ == '__main__':
    start_time = time.perf_counter()
    try:
        processed_list = Parallel(n_jobs=num_cores)(delayed(process_directory)(i) for i in list(f.keys()))
    except:
        pass
    finally:
        print(time.perf_counter() - start_time)

