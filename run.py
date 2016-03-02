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
from sklearn.multiclass import OneVsRestClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model

def load_sparse_csr(filename):
    loader = np.load(filename)
    return csr_matrix((  loader['data'], loader['indices'], loader['indptr']),
                         shape = loader['shape'])

def mean(l):
    return reduce(lambda x, y: x + y, l) / len(l)

algorithms = ["nb", "svm",  "lsvm", "lr"]
if len(sys.argv) > 1:
  algorithms = sys.argv[1:]

data_file_processed = './tmp/data.bigram.temp.npz'
target_file_processed = './tmp/target.bigram.temp.npy'

multiclass = True
print "Loading from file..."
X = load_sparse_csr(data_file_processed);
Y = np.load(target_file_processed)
print "File Loaded"

#split dataset to training set and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    X, Y, test_size=0.4, random_state=0)

#initialization
clfs = {}
if multiclass:
    clfs["nb"] = MultinomialNB()
    clfs["svm"] = OneVsRestClassifier(SVC())
    clfs["lsvm"] = OneVsRestClassifier(LinearSVC(random_state=0))
    clfs["lr"] = OneVsRestClassifier(linear_model.LogisticRegression(C=1e5))
else:
    clfs["nb"] = MultinomialNB()
    clfs["svm"] = SVC()
    clfs["lsvm"] = LinearSVC(random_state=0)
    clfs["lr"] = linear_model.LogisticRegression(C=1e5)

meta = {}
for a in algorithms:
    meta[a] = {}
    meta[a]["accuracy"] = []
    meta[a]["precision"] = []
    meta[a]["recall"] = []
    meta[a]["fscore"] = []

write_to_file = True
files = {}
for a in algorithms:
    filename = "./reports/" + a + ".unigram.report"
    files[a] = open(filename, "a+")

#cross validation against training dataset
skf = StratifiedKFold(y_train, n_folds=10, shuffle=True)
for train_index, test_index in skf:
    x, xt = X_train[train_index], X_train[test_index]
    y, yt = y_train[train_index], y_train[test_index]
    #test each algorithm
    for a in algorithms:
        clfs[a].fit(x, y)
        yp = clfs[a].predict(xt)
        meta[a]["report"] = metrics.classification_report(yt, yp);
        if write_to_file:
            files[a].write(meta[a]["report"])
        meta[a]["accuracy"].append(metrics.accuracy_score(yt, yp))
        meta[a]["precision"].append(metrics.precision_score(yt, yp, average='weighted'))
        meta[a]["recall"].append(metrics.recall_score(yt, yp, average='weighted'))
        meta[a]["fscore"].append(metrics.f1_score(yt, yp, average='weighted'))

#close file
for a in algorithms:
    files[a].close()

#print calssification result
for a in algorithms:
    print a
    print "*****"
    print "Accuracy:",
    print mean(meta[a]["accuracy"])
    print "Precison:",
    print mean(meta[a]["precision"])
    print "Recall:",
    print mean(meta[a]["recall"])
    print "F Score:",
    print mean(meta[a]["fscore"])


#run against test dataset
for a in algorithms:
    print a
    print "*****"
    clfs[a].fit(X_train, y_train)
    yp = clfs[a].predict(X_test)
    print metrics.classification_report(y_test, yp)
    print metrics.accuracy_score(y_test, yp)
    print metrics.precision_score(y_test, yp, average='weighted')
    print metrics.recall_score(y_test, yp, average='weighted')
    print metrics.f1_score(y_test, yp, average='weighted')
