import librosa
import samplerate
import wave
import numpy as np
ratio = 16000 / 44100
y, sr = librosa.load("output.wav", sr=44100)
print(y)
converter = 'sinc_best'
output_data_simple = samplerate.resample(y, ratio, converter)
print(output_data_simple)

wf = wave.open("output2.wav", 'wb')
wf.setnchannels(1)
wf.setsampwidth(1)
wf.setframerate(16000)
wf.writeframes(y.tobytes())
wf.close()
