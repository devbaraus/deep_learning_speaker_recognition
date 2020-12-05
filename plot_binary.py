import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import hann

audio1 = np.genfromtxt('./processed/p225.csv', delimiter=',')
audio2 = np.genfromtxt('./processed/p230.csv', delimiter=',')
audio3 = np.genfromtxt('./processed/p227.csv', delimiter=',')
audio4 = np.genfromtxt('./processed/p248.csv', delimiter=',')

audio_comp = audio4[0]
audios = np.asarray([audio1, audio2, audio3, audio4])


mse = np.sqrt(np.sum(audio_comp - audio4.mean(0)) ** 2)

print(mse)

# window = hann(1024)

# audio1 = audio1.mean(0) * window
# audio2 = audio2.mean(0) * window
# audio3 = audio3.mean(0) * window
# audio4 = audio4.mean(0) * window




plt.ylabel("Magnitude (dB)")
plt.xlabel("Frequency Bin")
plt.show()
