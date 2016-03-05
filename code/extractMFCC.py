
# coding: utf-8

# In[2]:

import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
import os


# In[74]:

def process(listpath, instrument, keyword, order = 1, hop_size = 1024, waittime = 1.8, note_duration = 1., note_frames = 10):
    
    mfcc_features = []
    
    with open(listpath, "r") as f:
        reached = False
        
        for line in f:
            
            # check whether it is target directory
            line = line.strip()
            if os.path.isdir(line):           
                
                head, tail = os.path.split(line)
                if tail.find(instrument)!=-1:
                    reached = True
                elif reached:
                    reached = False
                    break
                
                continue
            
            # not target directory
            if not reached:
                continue
            
            
            filename, file_extension = os.path.splitext(line)
            path, name = os.path.split(filename)
            if not (file_extension == ".aiff" or file_extension == ".aif"):
                continue
            
            #try to match key words
            if name.find(keyword)==-1:
                continue
            
            print line
            #load audio

            if not os.path.exists(line):
                continue
            sig,sr = librosa.load(line,sr = 44100)            
            
            # note onset dectection
            waitframe = waittime * sr / hop_size            
            onset_frames = librosa.onset.onset_detect(sig, sr=sr, hop_length = hop_size, delta=0.25, wait = waitframe)

            print onset_frames
            
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
                print feature.shape
                if feature.shape[1]!= note_frames:
                    continue
                mfcc_features.append(feature.flatten())
            
        print "Feature dimension", np.matrix(mfcc_features).shape
            
            
    return mfcc_features



if __name__ == "__main__":
    
    dstDir = os.path.join("../Features", "MFCC")
    if not os.path.exists(dstDir):
        os.mkdir(dstDir)
        print "Created new directroy", dstDir

    instruments = ['Sax', 'Flute','Violin','Piano', 'Guitar'] #['Violin', 'Guitar'] 
    keyword = '.ff.'
    orders = [1,2,3]
    
    
    
    listpath = 'D:\Files\Learning\Northwestern\EECS 352 Machine Perception of Music&Audio\project\Dataset\list.txt'
    for ins in instruments:
        for o in orders:
            print(ins, o)
            mfcc_features = process(listpath, ins, keyword, order = o)
            savename =  ins + keyword + "mfcc_dim10x"+str(o*13) + ".npz"
            savepath = os.path.join(dstDir, savename)
            np.savez(savepath, mfcc_features = mfcc_features)
            print "Saved"

    

