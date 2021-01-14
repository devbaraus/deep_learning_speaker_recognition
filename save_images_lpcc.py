from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import soundfile as sf
import spafe
import librosa
import librosa.display
from scipy.io.wavfile import read
from spafe.features.lpc import lpc, lpcc
from spafe.utils import vis

from utils import create_directory

num_cores = multiprocessing.cpu_count()

f = {}

# mypath = 'archive/VCTK-Corpus/VCTK-Corpus/wav48'
mypath = 'archive/hoogle'
destpath = 'images/hoogle'

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break

vmin = np.Inf
vmax = -np.Inf


def save_lpcc(sample, sample_rate, dest_path, audio_name, printimg=False):
    num_ceps = 13
    lifter = 0
    normalize = True
    # global vmin
    # global vmax

    # print(sample.T.shape)

    # intervals = librosa.effects.split(sample.astype(float).T, top_db=20)
    #
    # audio_temp = sample[intervals[0][0]:intervals[-1][-1]]

    lpcc_segment = lpcc(sig=sample, fs=sample_rate, num_ceps=num_ceps, lifter=lifter, normalize=normalize)

    # if np.min(lpcc_segment) < vmin:
    #     vmin = np.min(lpcc_segment)
    #
    # if np.max(lpcc_segment) > vmax:
    #     vmax = np.max(lpcc_segment)

    # if printimg:
    librosa.display.specshow(np.asarray(lpcc_segment).T, sr=sample_rate, x_axis='frames', y_axis='frames')
    plt.xlabel("Time (s)")
    plt.ylabel('Index')
    plt.title(f'LPCC - {audio_name}')
    plt.colorbar(format='%+2.0f')
    plt.savefig(f'{dest_path}.png')
    plt.close()


def process_directory(dir, index):
    for j, audioname in enumerate(f[dir]):
        if j < 5:
            # sr, sample = read(f'{mypath}/{dir}/{audioname}')
            sample, sr = librosa.load(f'{mypath}/{dir}/{audioname}')
            sample = sample.astype(np.int16)
            print(type(sample[0]))
            # sample = sample[:, 0]

            # if process_method == 'mfcc':
            create_directory(f'{destpath}/lpcc')
            save_lpcc(sample, sr, f'{destpath}/lpcc/{audioname.replace(".wav", "")}', audioname)

    # for j, audioname in enumerate(f[dir]):
    #     if j < 5:
    #         sample, sr = librosa.load(f'{mypath}/{dir}/{audioname}', sr=22050)
    #
    #         # if process_method == 'mfcc':
    #         save_lpcc(sample, sr, f'{destpath}/lpcc/{audioname.replace(".wav", "")}', audioname, True)
    #
    #         # # elif process_method == 'lpc':
    #         # save_lpc(sample, f'{destpath}/lpc/{audioname.replace(".wav", "")}', audioname)


if __name__ == '__main__':
    for j, i in enumerate(list(f.keys())):
        if j < 10:
            process_directory(i, j)

    # m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
    #     delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 10)
