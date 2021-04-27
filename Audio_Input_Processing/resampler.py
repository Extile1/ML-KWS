import librosa
import soundfile as sf
import shutil
import os

s = 16000
# inputfile = "off2.wav"
# outputfile = "test.wav"
#
# y, s = librosa.load(inputfile, sr=s)
# y = librosa.to_mono(y)
# sf.write(outputfile, y, s)

thisdir = os.getcwd()
fromreldir = "/../Audio_Input/Sixth_Person"
toreldir = "/../Audio_Input/Sixth_Person_Resample"
fromfulldir = thisdir + fromreldir
tofulldir = thisdir + toreldir
ignore = ["_background_noise_"]
for root, dirs, files, in os.walk(fromfulldir):
    print(root)
    print(len(files))
    folder = root.split("\\")[-1]
    print(folder)

    if len(files) > 10:
        if not os.path.exists(tofulldir + "/" + folder):
            os.mkdir(tofulldir + "/" + folder)

        for file in files:
            # print(root + file)
            y, s = librosa.load(root + "/" + file, sr=s)
            y = librosa.to_mono(y)
            sf.write(tofulldir + "/" + folder + "/" + file, y, s)
            shutil.copy(tofulldir + "/" + folder + "/" + file, thisdir + "/../Audio_Input/Six_People_Resample" + "/" + folder + "/" + file)