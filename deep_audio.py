import json
import os
from spafe.features.mfcc import mfcc
from spafe.features.lpc import lpcc, lpc
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.io.wavfile import read, write


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Visualization:

    @staticmethod
    def plot(data, title=None, x_label='Time (s)', y_label='Amplitude', size=(10, 6), caption=None,
             fig_name=None,
             show=False,
             close=True):

        if size:
            plt.figure(figsize=(10, 6), frameon=True)

        plt.plot(list(range(0, len(data))), data)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        # Remove a margem no eixo x
        plt.margins(x=0)

        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center')

        if fig_name:
            Directory.create_directory(fig_name, True)
            plt.savefig(fig_name, transparent=False)

        if show:
            plt.show()

        if close:
            plt.close()

    @staticmethod
    def plot_audio(data, rate, title=None, x_label='Time (s)', y_label='Amplitude', size=(10, 6), caption=None,
                   fig_name=None,
                   show=False,
                   close=True):

        time = np.linspace(0, len(data) / rate, num=len(data))
        if size:
            plt.figure(figsize=(10, 6), frameon=True)

        plt.plot(time, data)
        plt.ylabel(y_label)
        plt.xlabel(x_label)
        plt.title(title)
        # Remove a margem no eixo x
        plt.margins(x=0)

        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center')

        if fig_name:
            Directory.create_directory(fig_name, True)
            plt.savefig(fig_name, transparent=False)

        if show:
            plt.show()

        if close:
            plt.close()

    @staticmethod
    def plot_spectrogram(data, rate, n_fft=1024, title=None, x_label='Time (s)', y_label='Frequency (kHz)',
                         cmap='magma', size=(10, 6), caption=None, fig_name=None, show=False, close=True):

        if size:
            plt.figure(figsize=(10, 6), frameon=True)
        plt.specgram(data, NFFT=n_fft, Fs=rate, cmap=cmap)
        plt.title(title)
        plt.ylabel(y_label)
        plt.xlabel(x_label)

        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center')

        if fig_name:
            Directory.create_directory(fig_name, True)
            plt.savefig(fig_name, transparent=False)

        if show:
            plt.show()

        if close:
            plt.close()

    @staticmethod
    def plot_cepstrals(data, title=None, x_label='Frame Index', y_label='Index', cmap='magma', size=(10, 6),
                       caption=None,
                       fig_name=None,
                       show=False, close=True):

        if size:
            plt.figure(figsize=(10, 6), frameon=True)
        plt.imshow(data.T,
                   origin='lower',
                   aspect='auto',
                   cmap=cmap,
                   interpolation='nearest')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        # plt.colorbar(format='%+2.0f')
        # plt.clim(vmin, vmax)

        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center')

        if fig_name:
            Directory.create_directory(fig_name, True)
            plt.savefig(fig_name, transparent=False)

        if show:
            plt.show()

        if close:
            plt.close()

    @staticmethod
    def plot_subplots(audio, mfccs, lpccs, rate, title=None, size_multiplier=2, cmap='magma', caption=None,
                      fig_name=None, show=False):
        small_size = 8
        medium_size = 10
        bigger_size = 12

        image_size = (10 * size_multiplier, 6 * size_multiplier)

        # controls default text sizes
        plt.rc('font', size=small_size * size_multiplier)
        # fontsize of the axes title
        plt.rc('axes', titlesize=small_size * size_multiplier)
        # fontsize of the x and y labels
        plt.rc('axes', labelsize=medium_size * size_multiplier)
        # fontsize of the tick labels
        plt.rc('xtick', labelsize=small_size * size_multiplier)
        # fontsize of the tick labels
        plt.rc('ytick', labelsize=small_size * size_multiplier)
        plt.rc('legend', fontsize=small_size *
               size_multiplier)  # legend fontsize
        # fontsize of the figure title
        plt.rc('figure', titlesize=bigger_size * size_multiplier)

        plt.subplots(2, 2, figsize=image_size)
        plt.subplots_adjust(left=0.125, right=0.9, bottom=0.1,
                            top=0.9, wspace=0.3, hspace=0.3)
        plt.suptitle(title)

        plt.subplot(2, 2, 1)
        Visualization.plot_audio(audio, rate, close=False, size=None)

        plt.subplot(2, 2, 2)
        Visualization.plot_spectrogram(
            audio, rate, cmap=cmap, close=False, size=None)

        plt.subplot(2, 2, 3)
        Visualization.plot_cepstrals(
            data=lpccs, y_label='LPCC Index', cmap=cmap, size=None, close=False)

        plt.subplot(2, 2, 4)
        Visualization.plot_cepstrals(
            data=mfccs, y_label='MFCC Index', cmap=cmap, size=None, close=False)

        if caption:
            plt.figtext(0.5, 0.01, caption, wrap=True,
                        horizontalalignment='center')

        if fig_name:
            Directory.create_directory(fig_name, True)
            plt.savefig(fig_name, transparent=False)

        if show:
            plt.show()

        plt.close()

        # Reseta todo o estilo configurado no inicio da função
        mpl.rcParams.update(mpl.rcParamsDefault)


class Audio:
    @staticmethod
    def resample(data, current_rate, new_rate):
        import librosa

        data = librosa.resample(data, current_rate, new_rate)
        return data, new_rate

    @staticmethod
    def audioname(name):
        return name.replace('.wav', '')

    @staticmethod
    def read(path, sr=None, mono=True, normalize=True):
        import librosa

        data, rate = librosa.load(path, sr=sr, mono=mono)

        if not normalize:
            data = Audio.unnormalize(data)

        return data, rate

    @staticmethod
    def write(path, data, rate):
        import soundfile as sf

        Directory.create_directory(path, file=True)

        sf.write(path, data, rate, subtype='PCM_16')

    @staticmethod
    def to_mono(data):
        if len(data.shape) > 1 and data.shape[1] > 0:
            data = np.mean(data, axis=1, dtype=type(data[0][0]))
        return data

    @staticmethod
    def db(data, n_fft=2048):
        from librosa import stft, amplitude_to_db
        from numpy import abs, max
        S = stft(data, n_fft=n_fft, hop_length=n_fft // 2)
        D = amplitude_to_db(abs(S) * 1, max)
        return max(abs(D))

    @staticmethod
    def normalize(data):
        data_type = type(data[0])

        if data_type == np.int16:
            data = data.astype(np.float32) / 32768

        return data

    @staticmethod
    def unnormalize(data):
        data_type = type(data[0])

        if data_type == np.float32:
            data = np.array(data * 32768).astype(np.int16)

        return data

    @staticmethod
    def mfcc(data,
             rate=48000,
             n_mfcc=13,
             pre_emph=0,
             pre_emph_coeff=0.97,
             win_len=0.025,
             win_hop=0.01,
             win_type="hamming",
             n_filts=2048,
             n_fft=512,
             low_freq=None,
             high_freq=None,
             scale="constant",
             dct_type=2,
             use_energy=False,
             lifter=22,
             normalize=1):
        return mfcc(sig=data,
                    fs=rate,
                    num_ceps=n_mfcc,
                    pre_emph=pre_emph,
                    pre_emph_coeff=pre_emph_coeff,
                    win_len=win_len,
                    win_hop=win_hop,
                    win_type=win_type,
                    nfilts=n_filts,
                    nfft=n_fft,
                    low_freq=low_freq,
                    high_freq=high_freq,
                    scale=scale,
                    dct_type=dct_type,
                    use_energy=use_energy,
                    lifter=lifter,
                    normalize=normalize)

    @staticmethod
    def lpcc(data,
             rate=48000,
             n_lpcc=13,
             pre_emph=1,
             pre_emph_coeff=0.97,
             win_type="hann",
             win_len=0.025,
             win_hop=0.01,
             do_rasta=True,
             lifter=1,
             normalize=1,
             dither=1):
        return lpcc(sig=data,
                    fs=rate,
                    num_ceps=n_lpcc,
                    pre_emph=pre_emph,
                    pre_emph_coeff=pre_emph_coeff,
                    win_type=win_type,
                    win_len=win_len,
                    win_hop=win_hop,
                    do_rasta=do_rasta,
                    lifter=lifter,
                    normalize=normalize,
                    dither=dither)


class Model:
    @staticmethod
    def create_model_json_file(file, data):
        directory = '/'.join(file.split('/')[:-1])

        Directory.create_directory(directory)

        with open(file, "w") as json_file:
            json_file.write(data)

    @staticmethod
    def load_processed_data(path, inputs_fieldname='mfcc'):
        with open(path, 'r') as fp:
            data = json.load(fp)

        # convert lists into numpy arrays
        inputs = np.array(data[inputs_fieldname])
        targets = np.array(data['labels'])
        mapping = data['mapping']

        return inputs, targets, mapping


class Directory:
    @staticmethod
    def create_directory(directory, file=False):
        if file:
            directory = '/'.join(directory.split('/')[0:-1])
        try:
            os.makedirs(directory)
        except FileExistsError:
            pass

    @staticmethod
    def rename_directory(current, newname):
        os.rename(current, newname)

    @staticmethod
    def filenames(path):
        f = []

        for (_, _, filenames) in os.walk(path):
            f.extend(filenames)
            break

        return f

    @staticmethod
    def filenames_recursive(path):
        f = {}
        for (_, dirnames, _) in os.walk(path):
            for dir in dirnames:
                f[dir] = []
                for (_, _, filenames) in os.walk(path + '/' + dir):
                    f[dir].extend(filenames)
                    break
            break
        return f

    @staticmethod
    def load_json_data(path, inputs_fieldname):
        with open(path) as json_file:
            data = json.load(json_file)

            inputs = np.array(data[inputs_fieldname])
            labels = np.array(data['labels'])
            mapping = np.array(data['mapping'])

            return inputs, labels, mapping


class JSON:
    @staticmethod
    def create_json_file(file, data, indent=2):
        directory = '/'.join(file.split('/')[:-1])

        Directory.create_directory(directory)

        with open(file, "w") as fp:
            json.dump(data, fp, indent=indent, cls=NumpyEncoder)
