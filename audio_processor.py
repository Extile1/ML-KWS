
import pyaudio
import time
import numpy as np
import wave
import label_wav
import subprocess
import os


chunk = 1024  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
fs = 44100  # Record at 44100 samples per second
seconds = 2
filename = "output.wav"

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')

stream = p.open(format=sample_format,
                channels=channels,
                rate=fs,
                frames_per_buffer=chunk,
                input=True)

frames = []  # Initialize array to store frames

# Store data in chunks for 3 seconds
for i in range(0, int(fs / chunk * seconds)):
    data = stream.read(chunk)
    frames.append(data)

# Stop and close the stream
stream.stop_stream()
stream.close()
# Terminate the PortAudio interface
p.terminate()

print('Finished recording')

# Save the recorded data as a WAV file
wf = wave.open(filename, 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
wf.writeframes(b''.join(frames))
wf.close()

# test = subprocess.Popen(["python", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU\label_wav.py", "--wav", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU/"+filename, "--graph", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU\Pretrained_models\CRNN\CRNN_L.pb", "--labels", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU\Pretrained_models\labels.txt", "--how_many_labels", "3"], stdout=subprocess.PIPE)
# output = test.communicate()

# label_wav.start(wav = filename,
#           graph = "Pretrained_models/CRNN/CRNN_L.pb",
#           labels = "Pretrained_models/labels.txt",
#           how_many_labels = 3)