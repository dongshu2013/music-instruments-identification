# coding: utf-8

import numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
import os

def process(listpath, order = 1, hop_size = 1024, waittime = 1, note_duration = 1., note_frames = 10):
    mfcc_features = []
    with open(listpath, "r") as f:
        for line in f:
            # check whether it is target directory
            line = line.strip()
            ins, sr, filename = line.split('    ')
            sr = int(sr)
            ins = int(ins)
            print(filename)

            sig, sr = librosa.load(filename,sr = sr)
            # note onset dectection
            waitframe = waittime * sr / hop_size
            onset_frames = librosa.onset.onset_detect(sig, sr=sr, hop_length = hop_size, delta=0.2, wait = waitframe)
            print(onset_frames)

            n_frame = note_duration * sr // hop_size
            skip_frames = np.ceil(n_frame / note_frames)

            # generate mfcc feature
            mfcc = librosa.feature.mfcc(sig, sr=sr, n_mfcc=13, hop_length = hop_size)
            if order >= 2:
                mfcc_delta = librosa.feature.delta(mfcc)
                mfcc =  np.concatenate((mfcc, mfcc_delta))
            if order >= 3:
                mfcc_delta2 = librosa.feature.delta(mfcc_delta)
                mfcc =  np.concatenate((mfcc, mfcc_delta2))

            # select the onset feature for each note
            for onset in onset_frames:
                if onset + n_frame > mfcc.shape[1]:
                    continue
                feature =  mfcc[:,onset-skip_frames:onset+n_frame:skip_frames]
                if feature.shape[1] < note_frames:
                    continue
                elif feature.shape[1] > note_frames:
                    feature = feature[:,0:10]
                print(feature.shape)
                label = ins * np.ones(1)
                mfcc_features.append(np.append(label, feature.flatten()))
        print("Feature dimension", np.matrix(mfcc_features).shape)
    return mfcc_features

if __name__ == "__main__":
    dstDir = os.path.join("./mfcc")
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
        print("Created new directroy", dstDir)

    instruments = ['Sax', 'Flute','Violin','Piano', 'Guitar'] #['Violin', 'Guitar']
    orders = [1,2,3]

    listpath = '/Users/dongshu/Documents/listpath'
    for o in orders:
        mfcc_features = process(listpath, order = o)
        savename =  "mfcc_dim10x"+str(o*13) + ".test.npz"
        savepath = os.path.join(dstDir, savename)
        np.savez(savepath, mfcc_features = mfcc_features)
        print("Saved")

