
# coding: utf-8

# In[22]:

import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
import pickle, os
from sklearn.externals import joblib
from sklearn.svm import SVC, LinearSVC

# In[8]:
def preprocess():
    data = np.load("../Features/mfcc_note0.3s_patch/mfcc_dim10x39_training.npz")
    x_train, y_train = data['features'], data['labels']

    data = np.load("../Features/mfcc_note0.3s_patch/mfcc_dim10x39_test.npz")
    x_test, y_test = data['features'], data['labels']

    X = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test))

    mean_ = np.mean(X, axis = 0)
    std_ = np.std(X, axis = 0)
    
    np.savez('mfcc_dim10x39_stat.npz', mean=mean_, std = std_)

    classifier = sklearn.multiclass.OneVsRestClassifier(SVC(kernel='rbf', class_weight=None, probability=True))

    model =classifier.fit(znorm(x_train, mean_, std_), y_train)

    joblib.dump(model, './models/rbfSVM_dim390_model.pkl') 



def znorm(m, means, stds):
    ncols = len(means)
    #print ncols
    for i in range(ncols):
        #print i
        m[:,i] = (m[:,i] - means[i]) / stds[i]
    return m


def extract_mfcc(filepath, order = 1, hop_size = 1024, waittime = 0.2, note_duration = 0.3, note_frames = 10):
    
    mfcc_feature = []
    
                        
    filename, file_extension = os.path.splitext(filepath.strip())
    path, name = os.path.split(filename)
    if not (file_extension == ".mp3" or file_extension == ".wav" or file_extension == ".aif" or file_extension == ".aiff"):
        return None


    if not os.path.exists(filepath):
        print apath, "Error path"
        return None

    sig,sr = librosa.load(filepath,sr = 44100)
    
    # note onset dectection
    waitframe = waittime * sr / hop_size            
    onset_frames = librosa.onset.onset_detect(sig, sr=sr, hop_length = hop_size, delta=0.1, wait = waitframe)

    print onset_frames

    n_frame = note_duration * sr // hop_size
    skip_frames = np.ceil(n_frame / note_frames)

    lag = note_frames / 5

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
        #feature =  mfcc[:,onset-skip_frames:onset+n_frame:skip_frames]
        start_frame = onset - lag
        end_frame = start_frame + note_frames
        feature =  mfcc[:,start_frame:end_frame]

        if feature.shape[1]!= note_frames:
            continue
        mfcc_feature.append(feature.flatten())


    print "Feature dimension", np.matrix(mfcc_feature).shape         
            
    return np.array(mfcc_feature)


# In[108]:

def predict_instrument(filename, model, normfile=None, instruments = ['Sax', 'Piano', 'Violin', 'Flute','Guitar'], prob = True):
    mfcc = extract_mfcc(filename, order = 3, hop_size = 1024, waittime = 0.2, note_duration = 0.3, note_frames = 10)
    classfier = joblib.load(model)


    if normfile == None:
        
        stat = np.load(normfile)
        mean = stat['mean']
        std = stat['std']
        mfcc = znorm(mfcc, mean_, std_)
    else:
        mfcc = sklearn.preprocessing.scale(mfcc)
    
    if prob:
        prob = classifier.predict_proba(mfcc)
        mean_prob = np.mean(prob, axis = 0)

        label = instruments[np.argmax(mean_prob)]
        return mean_prob, label
    else:
        labels = classifier.predict(mfcc)
        counts = np.bincount(labels)
        label = instruments[np.argmax(counts)]
        return labels, label
    
    
if __name__ == "__main__":

      
    if len(sys.argv) <= 1:
        print "No Input Music File!"
        return None
    
    audio_path = sys.argv[1]
    
    if len(sys.argv) > 2:
        model = sys.argv[2]
    else:
        model = './models/rbfSVM_dim390_model.pkl'

    if len(sys.argv) > 3:
        normfile = sys.argv[3]
    else:
        normfile = None

    labels = ['Sax', 'Piano', 'Violin', 'Flute','Guitar']
    predicts, label = predict_instrument(audio_path, model, normfile, instruments= labels, prob=True)
    return label




