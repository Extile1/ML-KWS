import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write

chunk = 8000  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
record_fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output"

print(sd.query_devices())

# p = pyaudio.PyAudio()
#
# print("Recording")
#
# stream = p.open(format=sample_format,
#                 channels=channels,
#                 rate=record_fs,
#                 frames_per_buffer=chunk,
#                 input_device_index=1,
#                 input=True)
#
# frames = []  # Initialize array to store frames
# int_frames = []
#
# for i in range(0, int(record_fs / chunk * seconds)):
#     data = stream.read(chunk, exception_on_overflow=False)
#     frames.append(data)
#     data = np.fromstring(data, np.int16)
#     data = data.tolist()
#     int_frames.extend(data)
#
# stream.stop_stream()
# stream.close()
# p.terminate()
#
# print("Done")
#
# wf = wave.open(filename + ".wav", 'wb')
# wf.setnchannels(channels)
# wf.setsampwidth(p.get_sample_size(sample_format))
# wf.setframerate(record_fs)
# wf.writeframes(b''.join(frames))
# wf.close()
#
# plt.scatter(list(map(lambda x: x * record_fs / record_fs, range(len(int_frames)))), int_frames, s = [0.5] * len(int_frames))
# plt.show()

print("Recording")

myrecording = sd.rec(int(seconds * record_fs), samplerate=record_fs, channels=channels)
sd.wait()  # Wait until recording is finished
print(myrecording)
print(myrecording.shape)
write(filename + '.wav', record_fs, myrecording)  # Save as WAV file

plt.scatter(list(map(lambda x: x * record_fs / record_fs, range(len(myrecording)))), myrecording, s = [0.5] * len(myrecording))
plt.show()