from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyaudio
import time
import numpy as np
#import librosa
import wave
import subprocess
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import argparse
import sys
import warnings
from scipy import signal
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import tensorflow as tf
from matplotlib import pyplot as plt
import samplerate
import struct
import label_wav as lw
import time

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def load_graph(filename):
  """Unpersists graph from file as default graph."""
  with tf.gfile.FastGFile(filename, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')


def load_labels(filename):
  """Read in labels, one label per line."""
  return [line.rstrip() for line in tf.gfile.GFile(filename)]


def run_graph(wav_data, labels, input_layer_name, output_layer_name,
              num_top_predictions, times):
  times.append(time.clock())
  """Runs the audio data through the graph and prints predictions."""
  with tf.Session() as sess:
    # Feed the audio data as input to the graph.
    #   predictions  will contain a two-dimensional array, where one
    #   dimension represents the input image count, and the other has
    #   predictions per class
    softmax_tensor = sess.graph.get_tensor_by_name(output_layer_name)
    predictions, = sess.run(softmax_tensor, {input_layer_name: wav_data})

    # Sort to show labels in order of confidence
    top_k = predictions.argsort()[-num_top_predictions:][::-1]
    for node_id in top_k:
      human_string = labels[node_id]
      score = predictions[node_id]
      print('%s (score = %.5f)' % (human_string, score))
    times.append(time.clock())
    print(str(times[1] - times[0]) + " - Load audio")
    print(str(times[2] - times[1]) + " - Run model")
    #print(times[3] - times[2])
    return 0


def label_wav(wav, labels, graph, input_name, output_name, how_many_labels):
  times = [time.clock()]
  """Loads the model and labels, and runs the inference to print predictions."""
  if not wav or not tf.gfile.Exists(wav):
    tf.logging.fatal('Audio file does not exist %s', wav)

  if not labels or not tf.gfile.Exists(labels):
    tf.logging.fatal('Labels file does not exist %s', labels)

  if not graph or not tf.gfile.Exists(graph):
    tf.logging.fatal('Graph file does not exist %s', graph)

  labels_list = load_labels(labels)

  # load graph, which is stored in the default session
  load_graph(graph)

  with open(wav, 'rb') as wav_file:
    wav_data = wav_file.read()

  run_graph(wav_data, labels_list, input_name, output_name, how_many_labels, times)

chunk = 8000  # Record in chunks of 1024 samples
sample_format = pyaudio.paInt16  # 16 bits per sample
channels = 2
record_fs = 44100  # Record at 44100 samples per second
fs = 16000
seconds = 10
clip_length = 1
filename = "output"
min_volume = 500 #can change to 500 after normalizing
max_volume = 32768
window = 0.5 * 2 #min time between clips
min_sound_time = 0.005 #min time needed of consecutive noise
not_extreme_threshold = 0.25 #percentage of not extreme samples needed to be perceived as sound

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < threshold

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = []
    for i in snd_data:
        r.append(int(i*times))
    return r

p = pyaudio.PyAudio()  # Create an interface to PortAudio

print('Recording')
print("-------------------------")

frames = []  # Initialize array to store frames
index = 0
next_index = 0 #the next index outside the window
not_extreme = 0 #number of samples that are not above max volume or below min volume
extreme_list = []
extreme_list_2 = []

total = 0

count = 0
start = time.process_time()

def callback(in_data, frame_count, time_info, flag):
    global frames, total, index, not_extreme, count, next_index #global variables for filter coefficients and array
    times = [time.clock()]
    data = np.fromstring(in_data, np.int16)
    resampler = samplerate.Resampler()
    data = resampler.process(data, fs / record_fs)
    data = data.astype(np.int16)
    data = data.tolist()
    frames.extend(data)
    total += len(data)

    while True:
        if (index % 1000 == 0):
            print(not_extreme)
        if index + int(clip_length * fs / 2) > len(frames):
            break
        elif int(clip_length * fs / 2) > index:
            if max_volume > abs(frames[index]) > min_volume:
                not_extreme += 1
                # extreme_list.append(index)
            index += 1
        else:
            if max_volume > abs(frames[index]) > min_volume:
                not_extreme += 1
                # extreme_list.append(index)
            if max_volume > abs(frames[index - int(clip_length * fs / 2)]) and abs(
                    frames[index - int(clip_length * fs / 2)]) > min_volume:
                not_extreme -= 1
                # extreme_list_2.append(index - int(clip_length * fs / 2))

            if index >= next_index and not_extreme / (clip_length * fs / 2) > not_extreme_threshold:
                print(index)
                print(0.5 * index / fs)
                # print(frames[index - int(min_sound_time * fs):index+1])
                temp_frames = frames[index - int(clip_length * fs / 2): index + int(clip_length * fs / 2)]
                print(temp_frames)
                # plt.scatter(list(range(len(temp_frames))), temp_frames)
                # plt.show()
                next_index = index + int(window * fs)
                index += 1

                times.append(time.clock())

                wf = wave.open("Live_Input/" + filename + str(count) + ".wav", 'wb')
                wf.setnchannels(channels)
                wf.setsampwidth(p.get_sample_size(sample_format))
                wf.setframerate(fs)
                format = "%dh" % (len(temp_frames))
                wf.writeframes(struct.pack(format, *temp_frames))
                wf.close()

                times.append(time.clock())

                print(count)
                # os.system("python label_wav.py --wav Live_Input/" + filename + str(count) + ".wav --graph tmp/harry_debug/ten_words.pb --labels Pretrained_models\labels.txt --how_many_labels 3")
                label_wav("Live_Input/" + filename + str(count) + ".wav",
                          "Pretrained_models/labels.txt",
                          "tmp/harry_debug/ten_words.pb",
                          "wav_data:0",
                          "labels_softmax:0",
                          3)
                times.append(time.clock())
                # print(str(times[1] - times[0]) + " - Load audio")
                # print(str(times[2] - times[1]) + " - Save audio")
                # print(str(times[3] - times[2]) + " - Get label")
                # print(str(times[3] - times[0]) + " - Total")
                print("-------------------------")

                count += 1
            else:
                index += 1
    return (in_data, pyaudio.paContinue)


stream = p.open(format=sample_format,
                channels=channels,
                rate=record_fs,
                #frames_per_buffer=chunk,
                #input_device_index=1,
                input=True,
                stream_callback=callback)

print("Time: " + str(time.clock() - start))

stream.start_stream()

while stream.is_active():
    time.sleep(seconds)
    stream.stop_stream()
stream.close()

p.terminate()

plt.scatter(list(map(lambda x: x * record_fs / fs, range(len(frames)))), frames, s = [1] * len(frames))
plt.show()

print('Finished recording')
#print(extreme_list)
#print(extreme_list_2)

# Save the recorded data as a WAV file
wf = wave.open(filename + ".wav", 'wb')
wf.setnchannels(channels)
wf.setsampwidth(p.get_sample_size(sample_format))
wf.setframerate(fs)
format = "%dh" % (len(frames))
wf.writeframes(struct.pack(format, *frames))
wf.close()

#python D:\Pycharm Projects\KWS\KWS Base\ML-KWS\label_wav.py --wav D:\Pycharm Projects\KWS\KWS Base\ML-KWS/output.wav --graph D:\Pycharm Projects\KWS\KWS Base\ML-KWS/tmp/harry_debug/ten_words.pb --labels D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU\Pretrained_models\labels.txt --how_many_labels 3
#test = subprocess.Popen(["python", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS\label_wav.py", "--wav", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS/"+filename, "--graph", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS/tmp/harry_debug/ten_words.pb", "--labels", "D:\Pycharm Projects\KWS\KWS Base\ML-KWS-for-MCU\Pretrained_models\labels.txt", "--how_many_labels", "3"], stdout=subprocess.PIPE)
# test = subprocess.Popen(["python", "label_wav.py", "--wav", filename, "--graph", "tmp/harry_debug/ten_words.pb", "--labels", "Pretrained_models\labels.txt", "--how_many_labels", "3"], stdout=subprocess.PIPE)
# output = test.communicate()
# print(output)

#os.system("python label_wav.py --wav output.wav --graph tmp/harry_debug/ten_words.pb --labels Pretrained_models\labels.txt --how_many_labels 3")


# label_wav.start(wav = filename,
#           graph = "Pretrained_models/CRNN/CRNN_L.pb",
#           labels = "Pretrained_models/labels.txt",
#           how_many_labels = 3)

# pylint: disable=unused-import
from tensorflow.contrib.framework.python.ops import audio_ops as contrib_audio
# pylint: enable=unused-import
