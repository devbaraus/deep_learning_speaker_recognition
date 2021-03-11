from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import soundfile as sf
import librosa
import scipy

from deep_audio import Directory, Audio

num_cores = multiprocessing.cpu_count()

f = {}

mypath = './archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = f'audios'

# Directory.create_directory(f'{destpath}/treino')
# Directory.create_directory(f'{destpath}/inferencia')

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break


def process_directory(dir, n_rate):
    signal = []

    sr = 48000

    for j, audioname in enumerate(f[dir]):
        # if j < 10:
        holder_signal, sr = Audio.read(f'{mypath}/{dir}/{audioname}', sr=n_rate)

        intervals = librosa.effects.split(holder_signal, top_db=20)

        audio_temp = holder_signal[intervals[0][0]:intervals[-1][-1]]

        signal.extend(audio_temp)

    signal = np.array(signal)

    Audio.write(f'{destpath}/{n_rate}/{dir}.wav', signal, n_rate)


if __name__ == '__main__':
    # for j, i in enumerate(list(f.keys())):
    #     if j < 1:
    #         process_directory(i, j)
    m = Parallel(n_jobs=4, verbose=len(f.keys()))(
        delayed(process_directory)(i, rate) for j, i in enumerate(list(f.keys())) for rate in [16000, 22050])
