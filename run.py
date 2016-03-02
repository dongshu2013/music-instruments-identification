# -*- coding: utf-8 -*-
"""
Created on Mon Feb 01 14:42:11 2016

@author: Shu Dong
"""
import sys

import numpy as np
from scipy.sparse.csr import csr_matrix

from sklearn.cross_validation import StratifiedKFold
from sklearn import cross_validation
from sklearn import metrics

import argparse

parser = argparse.ArgumentParser(description='Used to training features.')
parser.add_argument('--feature', metavar='F', nargs=1, help='specify file containing feature matrix')
parser.add_argument('--label', metavar="L", nargs=1, help='specify file containing labels matrix')
args = parser.parse_args()


def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                       shape = loader['shape'])


def mean(l):
    return reduce(lambda x, y: x + y, l) / len(l)


def generate_train_test_set(feature, label):
    X = load_sparse_csr(feature);
    Y = np.load(label)
    X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)
    return (X_train, X_test, Y_train, Y_test)


def create_meta(model_names):
    meta = {}
    for a in model_names:
        meta[a] = {}
        meta[a]["accuracy"] = []
        meta[a]["precision"] = []
        meta[a]["recall"] = []
        meta[a]["fscore"] = []
    return meta


def train(models, meta, x_train, y_train):
    #cross validation against training dataset
    skf = StratifiedKFold(y_train, n_folds=10, shuffle=True)
    for train_index, test_index in skf:
        x, xt = x_train[train_index], x_train[test_index]
        y, yt = y_train[train_index], y_train[test_index]
        #test each algorithm
        for m in models.model_names():
	    model = models.get_model(m)
            model.fit(x, y)
            yp = model.predict(xt)
            meta[m]["report"] = metrics.classification_report(yt, yp);
            meta[m]["accuracy"].append(metrics.accuracy_score(yt, yp))
            meta[m]["precision"].append(metrics.precision_score(yt, yp, average='weighted'))
            meta[m]["recall"].append(metrics.recall_score(yt, yp, average='weighted'))
            meta[m]["fscore"].append(metrics.f1_score(yt, yp, average='weighted'))


def cpt_mean_score(meta):
    for m in meta.keys():
        print a
        print "*****"
        print "Accuracy:",
        print mean(meta[m]["accuracy"])
        print "Precison:",
        print mean(meta[m]["precision"])
        print "Recall:",
        print mean(meta[m]["recall"])
        print "F Score:",
        print mean(meta[m]["fscore"])


def test(models, x_test, y_test):
    for m in models.model_names():
	model = models.get_model(m)
        model.fit(x_train, y_train)
        yp = model.predict(y_test)
        print metrics.classification_report(y_test, yp)
        print metrics.accuracy_score(y_test, yp)
        print metrics.precision_score(y_test, yp, average='weighted')
        print metrics.recall_score(y_test, yp, average='weighted')
        print metrics.f1_score(y_test, yp, average='weighted')

def main():
    model_names = ["nb", "svm",  "lsvm", "lr"]
    model = Model(model_names)
    meta = create_meta(model_name)
    

if __name__ == "__main__":
    main()
