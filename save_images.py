from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import soundfile as sf
import librosa, librosa.display

num_cores = multiprocessing.cpu_count()

process_method = 'lpc'

f = {}

mypath = 'archive/VCTK-Corpus/VCTK-Corpus/wav48'
destpath = 'images'

for (_, dirnames, _) in walk(mypath):
    for dir in dirnames:
        f[dir] = []
        for (_, _, filenames) in walk(mypath + '/' + dir):
            f[dir].extend(filenames)
            break
    break


def save_lpc(sample, dest_path, audio_name):
    lpc = librosa.core.lpc(sample, 1024)
    lpc2 = librosa.core.lpc(sample, 4096)

    lpc = np.abs(lpc)
    lpc2 = np.abs(lpc2)

    plt.plot(lpc)
    plt.plot(lpc2)
    plt.legend(['ordem 1024', 'ordem 4096'])
    plt.title(f'LPC - {audio_name}')
    plt.xlabel('')
    plt.ylabel('LPC')

    plt.savefig(f'{dest_path}.png')
    plt.close()


def save_mfcc(sample, dest_path, audio_name):
    sr = 22050
    n_fft = 2048
    hop_length = 16

    mfcc_segment = librosa.feature.mfcc(sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

    librosa.display.specshow(mfcc_segment, sr=sr, hop_length=hop_length)
    plt.xlabel("Time")
    plt.ylabel('MFCC')
    plt.title(f'MFCC - {audio_name}')
    plt.savefig(f'{dest_path}.png')
    plt.close()


def process_directory(dir, index):
    for j, audioname in enumerate(f[dir]):
        if j < 10:
            sample, sr = librosa.load(f'{mypath}/{dir}/{audioname}', sr=22050)

            # if process_method == 'mfcc':
            save_mfcc(sample, f'{destpath}/mfcc/{audioname.replace(".wav", "")}', audioname)

            # elif process_method == 'lpc':
            save_lpc(sample, f'{destpath}/lpc/{audioname.replace(".wav", "")}', audioname)


if __name__ == '__main__':
    m = Parallel(n_jobs=num_cores, verbose=len(f.keys()), temp_folder='./tmp/')(
        delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 10)
