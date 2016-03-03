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
from models import Models

parser = argparse.ArgumentParser(description='Used to training features.')
parser.add_argument('--feature', metavar='F', nargs=1, help='specify file containing feature matrix')
parser.add_argument('--label', metavar="L", nargs=1, help='specify file containing labels matrix')
args = parser.parse_args()

FOLD = 10

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                       shape = loader['shape'])


def mean(l):
    return reduce(lambda x, y: x + y, l) / len(l)


def generate_train_test_set(feature, label):
    X = load_sparse_csr(feature);
    Y = np.load(label)
    return cross_validation.train_test_split(X, Y, test_size=0.4, random_state=0)


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
            meta[m]["report"] = metrics.classification_report(yt, yp)
            meta[m]["accuracy"] = metrics.accuracy_score(yt, yp)
            meta[m]["precision"] = metrics.precision_score(yt, yp, average='weighted')
            meta[m]["recall"] = metrics.recall_score(yt, yp, average='weighted')
            meta[m]["fscore"] = metrics.f1_score(yt, yp, average='weighted')
        train_result.append(meta)
    return train_result


def print_meta(meta):
    for m in meta.keys():
        print m
        print "****"
        print "Report:"
        print meta[m]['report'][i]
        print "Accuracy:"
        print meta[m]['accuracy'][i]
        print "Precision:"
        print meta[m]['precision'][i]
        print "Recall:"
        print meta[m]['recall'][i]
        print "F-Score:"
        print meta[m]['fscore'][i]


def test(models, x_train, y_train, x_test, y_test):
    test_result = {}
    for m in models.model_names():
        model = models.get_model(m)
        model.fit(x_train, y_train)
        yp = model.predict(y_test)
        test_result[m]["report"] = metrics.classification_report(y_test, yp)
        test_result[m]["accuracy"] = metrics.accuracy_score(y_test, yp)
        test_result[m]["precision"] = metrics.precision_score(y_test, yp, average='weighted')
        test_result[m]["recall"] = metrics.recall_score(y_test, yp, average='weighted')
        test_result[m]["fscore"] = metrics.f1_score(y_test, yp, average='weighted')
    return test_result


def main():
    x_train, x_test, y_train, y_test = generate_train_test_set(args['feature'], args['label'])
    model_names = ["nb", "svm",  "lsvm", "lr"]
    model = Model(model_names)
    train_result = train(modle, x_train, y_train)
    for meta in train_result:
        print_meta(meta)


if __name__ == "__main__":
    main()
