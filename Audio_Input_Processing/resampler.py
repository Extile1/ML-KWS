import librosa
import soundfile as sf
import shutil
import os

s = 16000
inputfile = "off2.wav"
outputfile = "test.wav"

y, s = librosa.load(inputfile, sr=s)
y = librosa.to_mono(y)
sf.write(outputfile, y, s)

thisdir = os.getcwd()
fromreldir = "/../Audio_Input/Google_Train"
toreldir = "/../Audio_Input/Google_Test"
fromfulldir = thisdir + fromreldir
tofulldir = thisdir + toreldir
testpercent = 0.05
ignore = ["_background_noise_"]
for root, dirs, files, in os.walk(fromfulldir):
    print(root)
    print(len(files))
    folder = root.split("\\")[-1]
    if (len(files) > 10):
        numtesting = math.floor(len(files) * testpercent)
        print(numtesting)
        for file in files[:numtesting]:
            # print(root + file)
            shutil.move(root + "/" + file, tofulldir + "/" + folder + "/" + file)