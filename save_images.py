from joblib import Parallel, delayed
import multiprocessing
from deep_audio import Directory, Audio, Visualization

num_cores = multiprocessing.cpu_count()

origin_path = 'archive/VCTK-Corpus/VCTK-Corpus/wav48'
# origin_path = 'archive/hoogle'


f = Directory.filenames_recursive(origin_path)


def process_directory(dir, index):
    n_mfcc = 40
    n_filts = 512
    n_fft = 2048

    n_lpcc = 27
    lifter = 0
    normalize = False

    for _, audio in enumerate(f[dir]):
        audioname = Audio.audioname(audio)

        data, rate = Audio.read(f'{origin_path}/{dir}/{audio}')

        mfcc = Audio.mfcc(data, rate, n_mfcc=n_mfcc, n_filts=n_filts, n_fft=n_fft)
        lpcc = Audio.lpcc(data, rate, n_lpcc=n_lpcc, lifter=lifter, normalize=normalize)

        Visualization.plot_audio(data, rate, title=f'Signal {audioname}',
                                 fig_name=f'images/audio/{dir}/{audioname}.png',
                                 close=True)
        Visualization.plot_spectrogram(data, rate, title=f'Spectrogram {audioname}',
                                       fig_name=f'images/spectrogram/{dir}/{audioname}.png', close=True)
        Visualization.plot_cepstrals(mfcc, title=f'MFCC {audioname}', fig_name=f'images/mfcc/{dir}/{audioname}.png',
                                     y_label='MFCC Index', close=True)
        Visualization.plot_cepstrals(lpcc, title=f'LPCC {audioname}', fig_name=f'images/lpcc/{dir}/{audioname}.png',
                                     y_label='LPCC Index', close=True)

        Visualization.plot_subplots(audio=data, mfccs=mfcc, lpccs=lpcc, rate=rate, title=f'{audioname}',
                                    fig_name=f'images/plots/{dir}/{audioname}.png', close=True)


if __name__ == '__main__':
    # for j, i in enumerate(list(f.keys())):
    #     if j < 2:
    #         process_directory(i, j)

    m = Parallel(n_jobs=num_cores, verbose=len(f.keys()))(
        delayed(process_directory)(i, j) for j, i in enumerate(list(f.keys())))
