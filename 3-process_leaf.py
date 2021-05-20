#!/usr/bin/env python
# coding: utf-8

# In[1]:


from joblib import Parallel, delayed
import numpy as np
import multiprocessing
from tensorflow import newaxis
import leaf_audio.frontend as frontend
from deep_audio import Directory, JSON, Audio, NumpyEncoder


# In[2]:


num_cores = multiprocessing.cpu_count()

sampling_rate = 24000

path = f'audios/{sampling_rate}'

f = Directory.filenames(path)


# In[3]:


def process_directory(dir, index, library):
    signal, rate = Audio.read(
        f'{path}/{dir}', sr=sampling_rate, normalize=True)

    signal = np.array(signal)

    signal = signal[:len(signal) - len(signal) % (rate * 5)]

    segments = len(signal) // (rate * 5)

    m = {
        "mfcc": [],
        "labels": [index] * segments
    }

    for i in range(segments):
        start_sample = rate * i * 5
        finish_sample = start_sample + (rate * 5)

        sample = signal[start_sample:finish_sample]
        sample = sample[newaxis, :]

        if library == 'leaf':
            leaf = frontend.Leaf()
            mfcc = leaf(sample)
        elif library == 'melbanks':
            melfbanks = frontend.MelFilterbanks()
            mfcc = melfbanks(sample)
        elif library == 'tfbanks':
            tfbanks = frontend.TimeDomainFilterbanks()
            mfcc = tfbanks(sample)
        elif library == 'sincnet':
            sincnet = frontend.SincNet()
            mfcc = sincnet(sample)
        elif library == 'sincnetplus':
            sincnet_plus = frontend.SincNetPlus()
            mfcc = sincnet_plus(sample)

        mfcc = np.array(mfcc).T

        m['mfcc'].append(mfcc.tolist())

        del mfcc
        del sample

    print(f'{dir} -> segments: {segments}')
    return m


# In[4]:


def object_mfcc_to_json(m, library):
    data = {
        "mapping": [],
        "mfcc": [],
        "labels": []
    }

    data['mapping'] = [file.replace('.wav', '') for i, file in enumerate(f)]

    for i in m:
        data['mfcc'].extend(i['mfcc'])
        data['labels'].extend(i['labels'])

    print('Writing')

    JSON.create_json_file(
        f'processed/leaf/{library}_{sampling_rate}.json', data, cls=NumpyEncoder)

    del data


# In[ ]:


if __name__ == '__main__':
    #     for library in ['leaf', 'melbanks', 'tfbanks', 'sincnet', 'sincnetplus']:
    #         m = []
    #         for j, i in enumerate(f):
    #             if j  < 5:
    #                 m.append(process_directory(i, j, library))

    #         object_mfcc_to_json(m, library)

    for library in ['melbanks', 'tfbanks', 'sincnet', 'sincnetplus']:
        m = Parallel(n_jobs=num_cores, verbose=len(f))(
            delayed(process_directory)(i, j, library) for j, i in enumerate(f) if j < 4)
        object_mfcc_to_json(m, library)
        del m


# In[ ]:


# In[ ]:
