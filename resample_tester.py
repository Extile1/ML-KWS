import librosa
import samplerate
import wave
import numpy as np
from matplotlib import pyplot as plt

y, sr = librosa.load("unsampled.wav", sr=16000)
# ratio = 16000 / sr
# print(y)
# converter = 'sinc_best'
# output_data_simple = samplerate.resample(y, ratio, converter)
# print(output_data_simple)

plt.scatter(list(range(len(y))), y, s = [4] * len(y))
# #data = librosa.resample(data, record_fs, fs, res_type='linear')
# resampler = samplerate.Resampler()
# data = resampler.process(data, fs / record_fs)
# data = signal.resample(data, int(data.size * fs / record_fs))
# #data = data.tobytes()
# data *= 1 / np.max(np.abs(data),axis=0)
# data += 1
# data *= 255.0 / 2
# data = data.astype(np.int16)
# data = np.clip(data, 0, 255)
# data = data.tolist()
# print(data)
# plt.scatter(list(map(lambda x: x * record_fs / fs, range(len(data)))), data, s = [4] * len(data))
plt.show()
print(y)

# wf = wave.open("output2.wav", 'wb')
# wf.setnchannels(1)
# wf.setsampwidth(1)
# wf.setframerate(16000)
# wf.writeframes(y.tobytes())
# wf.close()
