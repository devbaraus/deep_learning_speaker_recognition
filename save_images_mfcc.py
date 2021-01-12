from os import walk
from joblib import Parallel, delayed
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from spafe.utils import vis
import soundfile as sf
import librosa, librosa.display

from utils import create_directory

num_cores = multiprocessing.cpu_count()

process_method = 'lpc'

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


def save_mfcc(sample, dest_path, audio_name, printimg=False):
    sr = 22050
    n_fft = 2048
    hop_length = 512
    n_mfcc = 13
    global vmin
    global vmax

    # sample = librosa.amplitude_to_db(librosa.stft(sample), ref=np.max)

    # intervals = librosa.effects.split(sample, top_db=20)
    #
    # audio_temp = sample[intervals[0][0]:intervals[-1][-1]]

    mfcc_segment = librosa.feature.mfcc(sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mfcc=n_mfcc)

    if np.min(mfcc_segment) < vmin:
        vmin = np.min(mfcc_segment)

    if np.max(mfcc_segment) > vmax:
        vmax = np.max(mfcc_segment)

    if printimg:
        plt.specgram(sample, NFFT=1024, Fs=sr)
        plt.title(f'Spec - {audio_name}')
        plt.ylabel("Frequency (kHz)")
        plt.xlabel("Time (s)")
        plt.savefig(f'images/hoogle/spec/{audio_name}.png')
        # plt.show(block=False)
        plt.close()
        librosa.display.specshow(mfcc_segment, sr=sr, hop_length=hop_length, x_axis='s', y_axis='frames')
        plt.xlabel("Time (s)")
        plt.ylabel('Index')
        plt.title(f'MFCC - {audio_name}')
        plt.colorbar(format='%+2.0f')
        plt.clim(vmin, vmax)
        plt.savefig(f'{dest_path}_{n_mfcc}.png')
        # plt.show()
        plt.close()


        # for segment in range(0, 13):
        #     plt.plot(mfcc_segment[segment,:])
        # plt.xlabel("Time")
        # plt.ylabel('MFCC')
        # plt.title(f'MFCC - {audio_name}')
        # plt.savefig(f'{dest_path}_{hop_length}_segment.png')
        # # plt.close()
        # for segment in range(0, mfcc_segment.shape[1]):
        #     plt.plot(mfcc_segment[:, segment])
        # plt.xlabel("Time")
        # plt.ylabel('MFCC')
        # plt.title(f'MFCC - {audio_name}')
        # plt.savefig(f'{dest_path}_{hop_length}_time.png')
        # plt.close()


def process_directory(dir, index):
    for j, audioname in enumerate(f[dir]):
        if j < 5:
            sample, sr = librosa.load(f'{mypath}/{dir}/{audioname}', sr=22050)

            # if process_method == 'mfcc':
            create_directory(f'{destpath}/mfcc')
            save_mfcc(sample, f'{destpath}/mfcc/{audioname.replace(".wav", "")}', audioname)

    for j, audioname in enumerate(f[dir]):
        if j < 5:
            sample, sr = librosa.load(f'{mypath}/{dir}/{audioname}', sr=22050)

            # if process_method == 'mfcc':
            create_directory(f'{destpath}/mfcc')
            save_mfcc(sample, f'{destpath}/mfcc/{audioname.replace(".wav", "")}', audioname, True)

            # # elif process_method == 'lpc':
            # save_lpc(sample, f'{destpath}/lpc/{audioname.replace(".wav", "")}', audioname)


if __name__ == '__main__':
    for j, i in enumerate(list(f.keys())):
        if j < 10:
            process_directory(i, j)

    # m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
    #     delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())) if j < 10)
