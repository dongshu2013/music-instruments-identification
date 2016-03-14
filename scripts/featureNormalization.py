import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
import os


def znorm(m, means, stds):
    ncols = len(means)
    for i in range(ncols):
        m[:,i] = (m[:,i] - means[i]) / stds[i]
    return m


def patch_train_feature(datatype, dim = '10x13', norm = True):
    instruments = ['Sax', 'Piano', 'Violin', 'Flute','Guitar']
    rootDir = 'D:\Files\Learning\Northwestern\EECS 352 Machine Perception of Music&Audio\project\Features\MFCC'
    saveDir = 'D:\Files\Learning\Northwestern\EECS 352 Machine Perception of Music&Audio\project\Features'


    a,b = dim.split('x')
    d = int(a) * int(b)
    features = np.array([]).reshape(0,d)
    labels = []

    for i in range(5):
        ins = instruments[i]
        filename = ins + datatype + "mfcc_dim" + dim + ".npz"
        print filename

        filepath = os.path.join(rootDir, filename)

        data = np.load(filepath)
        mfcc = data['mfcc_features']
        r, d = mfcc.shape

        labels += [i] * r

        features = np.vstack([features, mfcc])

    if norm:
        scaler = sklearn.preprocessing.StandardScaler().fit(features)
        means = scaler.mean_
        stds = scaler.scale_
        features = scaler.transform(features)

        np.savez(os.path.join(saveDir,'mfcc_dim'+ dim + '_stat.npz'), mean=means, std = stds)
        savename = 'mfcc_dim'+ dim + '_training_norm.npz'
    else:
        savename = 'mfcc_dim'+ dim + '_training.npz'

    savepath = os.path.join(saveDir, savename)
    np.savez(savepath, features = features, labels = labels)
    print "Finished"

    if norm:
        return features, means, stds
    else:
        return features

def patch_test_feature(datatype, dim = '10x13', norm = True, stat_file=None):
    instruments = ['Sax', 'Piano', 'Violin', 'Flute','Guitar']
    rootDir = './'
    saveDir = './'

    a,b = dim.split('x')
    d = int(a) * int(b)
    features = np.array([]).reshape(0,d)
    labels = []

    for i in range(5):
        ins = instruments[i]
        filename = ins + datatype + "mfcc_dim" + dim + ".npz"
        print filename

        filepath = os.path.join(rootDir, filename)

        data = np.load(filepath)
        mfcc = data['mfcc_features']
        r, d = mfcc.shape

        labels += [i] * r

        features = np.vstack([features, mfcc])

    if norm:
        if stat_file!=None:
            stat = np.load(stat_file)
            means = stat['mean']
            stds = stat['std']
            features = np.array(znorm(features, means, stds))

        else:
            features = sklearn.preprocessing.scale(features)

        savename = 'mfcc_dim'+ dim + '_test_norm.npz'
    else:
        savename = 'mfcc_dim'+ dim + '_test.npz'

    savepath = os.path.join(saveDir, savename)
    np.savez(savepath, features = features, labels = labels)
    print "Finished"

    return features, labels

