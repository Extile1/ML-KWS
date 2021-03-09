import winsound
import os
import time

thisdir = os.getcwd()
fromreldir = "/../Audio_Input/Google_Test"
fromfulldir = thisdir + fromreldir
num_files = 10
keywords = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]

for root,dirs,files, in os.walk(fromfulldir):
    print(root)
    folder = root.split("\\")[-1]

    if (len(files) > 1 and (folder in keywords)):
        for n in range(num_files):
            #print(root + file)
            filename = (fromreldir + "/" + root.split("\\")[-1] + "/" + files[n])[1:]
            #print(filename)
            winsound.PlaySound(filename, winsound.SND_FILENAME)
            time.sleep(1)


#filename = "../Audio_Input/Google_Test/down/00b01445_nohash_0"
