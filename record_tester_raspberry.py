import pyaudio
import wave
import numpy as np
from matplotlib import pyplot as plt
import sounddevice as sd
from scipy.io.wavfile import write
import time
import struct

chunk = 8000  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
record_fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output_raspberry_3"

# p = pyaudio.PyAudio()
#
# print("Recording")
#
# stream = p.open(format=sample_format,
#                 channels=channels,
#                 rate=record_fs,
#                 frames_per_buffer=chunk,
#                 #input_device_index=1,
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

# myrecording = sd.rec(int(seconds * record_fs), samplerate=record_fs, channels=channels)
# sd.wait()  # Wait until recording is finished
# write(filename + '.wav', record_fs, myrecording)  # Save as WAV file

# plt.scatter(list(map(lambda x: x * record_fs / record_fs, range(len(myrecording)))), myrecording, s = [0.5] * len(myrecording))
# plt.show()

p = pyaudio.PyAudio()

frames = []
index = 0
next_index = 0 #the next index outside the window

def callback(in_data, frame_count, time_info, flag):
    global frames, total, index, not_extreme, count, next_index, length_1, length_2 #global variables for filter coefficients and array\
    length_1 += frame_count
    data = np.fromstring(in_data, np.int16)
    data = data.tolist()
    frames.extend(data)

    if length_1 >= record_fs * seconds * channels:
        return in_data, pyaudio.paComplete
    else:
        return in_data, pyaudio.paContinue

stream = p.open(format=sample_format,
                channels=channels,
                rate=record_fs,
                #frames_per_buffer=chunk,
                #input_device_index=1,
                input=True,
                stream_callback=callback)

stream.start_stream()

while stream.is_active():
    time.sleep(0.1)
stream.close()

p.terminate()

print('Finished recording')

plt.scatter(list(map(lambda x: x * record_fs / record_fs, range(len(frames)))), frames, s = [1] * len(frames))
plt.show()
# Save the recorded data as a WAV file
wf = wave.open(filename + ".wav", 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(record_fs)
format = "%dh" % (len(frames))
wf.writeframes(struct.pack(format, *frames))
wf.close()