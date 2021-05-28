import matplotlib.pyplot as plt
from deep_audio import Directory


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


def plot_audio(data, rate, title=None, x_label='Time (s)', y_label='Amplitude', size=(10, 6), caption=None,
               fig_name=None,
               show=False,
               close=True):
    from numpy import linspace

    time = linspace(0, len(data) / rate, num=len(data))
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


def plot_subplots(audio, mfccs, lpccs, rate, title=None, size_multiplier=2, cmap='magma', caption=None,
                  fig_name=None, show=False):
    from matplotlib import rcParams, rcParamsDefault

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
    plot_audio(audio, rate, close=False, size=None)

    plt.subplot(2, 2, 2)
    plot_spectrogram(
        audio, rate, cmap=cmap, close=False, size=None)

    plt.subplot(2, 2, 3)
    plot_cepstrals(
        data=lpccs, y_label='LPCC Index', cmap=cmap, size=None, close=False)

    plt.subplot(2, 2, 4)
    plot_cepstrals(
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
    rcParams.update(rcParamsDefault)
