import sys

import numpy as np
from scipy.sparse.csr import csr_matrix

from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn import metrics
from sklearn.preprocessing import Normalizer, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

import argparse
from models import Models
from functools import reduce
from load_features import load_train, load_test

#parser = argparse.ArgumentParser(description='Used to training features.')
#parser.add_argument('--feature', metavar='F', nargs=1, help='specify file containing feature matrix')
#parser.add_argument('--label', metavar="L", nargs=1, help='specify file containing labels matrix')
#args = parser.parse_args()

FOLD = 10

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
            shape = loader['shape'])


def mean(l):
    return reduce(lambda x, y: x + y, l) / len(l)


#def generate_train_test_set(feature, label):
#    X = load_sparse_csr(feature);
#    Y = np.load(label)
#    return cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)

def generate_train_test_set(X, Y):
    return cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)


def train(models, x_train, y_train):
    train_result = []
    #cross validation against training dataset
    skf = StratifiedKFold(y_train, n_folds=FOLD, shuffle=True)
    for train_index, test_index in skf:
        x, xt = x_train[train_index], x_train[test_index]
        y, yt = y_train[train_index], y_train[test_index]
        #test each algorithm
        meta = {}
        for m in models.model_names():
            model = models.get_model(m)
            model.fit(x, y)
            yp = model.predict(xt)
            meta[m] = {}
            meta[m]["report"] = metrics.classification_report(yt, yp)
            meta[m]["accuracy"] = metrics.accuracy_score(yt, yp)
            meta[m]["precision"] = metrics.precision_score(yt, yp, average='weighted')
            meta[m]["recall"] = metrics.recall_score(yt, yp, average='weighted')
            meta[m]["fscore"] = metrics.f1_score(yt, yp, average='weighted')
        train_result.append(meta)
    return train_result

def print_train_result(train_result):
    mean = {}
    for meta in train_result:
        print("------------------------------")
        for m in meta.keys():
            mean.setdefault(m, {})
            mean[m].setdefault('accuracy', [])
            mean[m].setdefault('precision', [])
            mean[m].setdefault('recall', [])
            mean[m].setdefault('fscore', [])
            print(m)
            print("****")
            print("Report:")
            print(meta[m]['report'])
            print("Accuracy:")
            mean[m]['accuracy'].append(meta[m]['accuracy'])
            print(meta[m]['accuracy'])
            print("Precision:")
            mean[m]['precision'].append(meta[m]['precision'])
            print(meta[m]['precision'])
            print("Recall:")
            mean[m]['recall'].append(meta[m]['recall'])
            print(meta[m]['recall'])
            print("F-Score:")
            mean[m]['fscore'].append(meta[m]['fscore'])
            print(meta[m]['fscore'])
        print("------------------------------")
    print_mean(mean)


def print_test_result(meta):
    for m in meta.keys():
        print(m)
        print("****")
        print("Report:")
        print(meta[m]['report'])
        print("Accuracy:")
        print(meta[m]['accuracy'])
        print("Precision:")
        print(meta[m]['precision'])
        print("Recall:")
        print(meta[m]['recall'])
        print("F-Score:")
        print(meta[m]['fscore'])


def print_mean(m):
    for key, value in m.items():
        for metric, seq in value.items():
            print(key + "." + metric + ":" + str(mean(seq)))


def test(models, x_train, y_train, x_test, y_test):
    test_result = {}
    for m in models.model_names():
        test_result[m] = {}
        model = models.get_model(m)
        model.fit(x_train, y_train)
        yp = model.predict(x_test)
        test_result[m]["report"] = metrics.classification_report(y_test, yp)
        test_result[m]["accuracy"] = metrics.accuracy_score(y_test, yp)
        test_result[m]["precision"] = metrics.precision_score(y_test, yp, average='weighted')
        test_result[m]["recall"] = metrics.recall_score(y_test, yp, average='weighted')
        test_result[m]["fscore"] = metrics.f1_score(y_test, yp, average='weighted')
    return test_result


def run(models, x_train, x_test, y_train, y_test):
    train_result = train(models, x_train, y_train)
    test_result = test(models, x_train, y_train, x_test, y_test)
    print_test_result(test_result)
#    for meta in train_result:
#        print "------------------------------"
#        print_meta(meta)
#        print "------------------------------"

def run2(x_train, x_test, y_train, y_test, models):
    print("xxx")


def feature_label(features):
    labels = features[:,0].reshape(-1)
    return normalize(features[:,1:]), labels


def normalize(features):
    nm = Normalizer()
    min_max_scaler = MinMaxScaler(feature_range=(0, 10))
    features = nm.fit_transform(features)
    return min_max_scaler.fit_transform(features)


def main():
    model_names = ["nb", "svm",  "lsvm", "lr", "psvm"]
    models = Models(model_names)

    data = np.load("../feature/mfcc_note0.3s_patch/mfcc_dim10x39_training.npz")
    x_train = np.array(data['features'])
    y_train = np.array(data['labels'])

    data = np.load("../feature/mfcc_note0.3s_patch/mfcc_dim10x39_test.npz")
    x_test = np.array(data['features'])
    y_test = np.array(data['labels'])

    feature = np.concatenate((x_train, x_test));
    feature = normalize(feature)

    #PCA
    #pca = PCA(n_components = 390)
    #feature = pca.fit_transform(feature)
    #feature = normalize(feature)

    x_train = feature[:x_train.shape[0]]
    x_test = feature[x_train.shape[0]:]

    print x_train.shape
    print x_test.shape
#
#    labels = np.concatenate((y_train, y_test))
#    for l in labels:
#        print(l)
#
    #LDA
    #features = np.concatenate((x_train, x_test))
    #lda = LinearDiscriminantAnalysis()
    #features = normalize(lda.fit_transform(features, labels))

#    x_train, y_train = feature_label(load_train())
#    x_test, y_test = feature_label(load_test())
    test_result = test(models, x_train, y_train, x_test, y_test)
    print_test_result(test_result)

#    train_result = train(models, features, labels)
#    print_train_result(train_result)

if __name__ == "__main__":
    main()
