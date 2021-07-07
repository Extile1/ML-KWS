import pyaudio
import wave

chunk = 8000  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 1
record_fs = 44100  # Record at 44100 samples per second
seconds = 3
filename = "output"

p = pyaudio.PyAudio()

print("Recording")

stream = p.open(format=sample_format,
                channels=channels,
                rate=record_fs,
                frames_per_buffer=chunk,
                #input_device_index=1,
                input=True)

frames = []  # Initialize array to store frames


for i in range(0, int(record_fs / chunk * seconds)):
    data = stream.read(chunk, exception_on_overflow=False)
    frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

print("Done")

wf = wave.open(filename + ".wav", 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(record_fs)
# format = "%dh" % (len(frames))
# wf.writeframes(struct.pack(format, *frames))
wf.writeframes(b''.join(frames))
wf.close()