import shutil
import os
import math

thisdir = os.getcwd()
fromreldir = "/../Audio_Input/Google_Train"
toreldir = "/../Audio_Input/Google_Test"
fromfulldir = thisdir + fromreldir
tofulldir = thisdir + toreldir
testpercent = 0.05
ignore = ["_background_noise_"]
undo = False

#print(list(os.walk(fulldir)))

if (undo):
    for root,dirs,files, in os.walk(tofulldir):
        print(root)
        print(len(files))
        folder = root.split("\\")[-1]

        if (len(files) > 0 and not (folder in ignore)):
            numtesting = math.floor(len(files) * 1)
            print(numtesting)
            for file in files[:numtesting]:
                #print(root + file)
                shutil.move(root + "/" + file, fromfulldir + "/" + folder + "/" + file)
else:
    for root,dirs,files, in os.walk(fromfulldir):
        print(root)
        print(len(files))
        folder = root.split("\\")[-1]
        if (len(files) > 10):
            numtesting = math.floor(len(files) * testpercent)
            print(numtesting)
            for file in files[:numtesting]:
                #print(root + file)
                shutil.move(root + "/" + file, tofulldir + "/" + folder + "/" + file)






