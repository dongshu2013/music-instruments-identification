import json
import re
import string
import os.path

import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder

def load_stop_words():
    return open('./stop_words').readlines()

def remove_punc(token):
    return "".join(c for c in token if c not in string.punctuation)

def clean_tokens(tokens):
    return [remove_punc(x) for x in tokens if not any(c.isdigit() for c in x)]

REGEX = re.compile(r"[,\n\s\\\/]*")
def tokenize(text):
    tokens = [tok.strip().lower() for tok in REGEX.split(text)]
    return clean_tokens(tokens)

def save_sparse_csr(filename, array):
    np.savez(filename,data = array.data ,indices=array.indices,
             indptr =array.indptr, shape=array.shape )

def prepare_raw_data(target_file, data_file):
    #load target
    f = open(target_file)
    fin2target = {}
    for line in f.readlines():
        val = line.strip().split(",")
        if len(val) == 2:
            #fin2target[str(val[0])] = val[1]
            if val[1] != 'green' and val[1] != 'orange':
                fin2target[str(val[0])] = 'other'
            else:
                fin2target[str(val[0])] = val[1]

    #extract features from file
    contents = []
    targets = []
    f = open(data_file)
    for line in f.readlines():
        profile = json.loads(line)
        fin = str(profile['_id'])
        if fin in fin2target:
            notes = [str(note['contents_plain_txt']) for note in profile['bcvi_large_01052016']]
            unit = note["current_unit"]
            contents.append(' '.join(notes) + ' ' + unit)
            targets.append(fin2target[fin])

    return contents, targets

def transform(data, labels):
    #normalize training data
    tfidf_vector = TfidfVectorizer(stop_words='english',ngram_range=(2,2),max_df=0.8,min_df=0.2,max_features=10000)
    X_train_tfidf = tfidf_vector.fit_transform(data)

    with open('./tmp/features.txt', 'w') as f:
    	for feature in tfidf_vector.get_feature_names():
	    f.write(feature)
	    f.write("\n")

   #encode label
    le = LabelEncoder()
    Y_labels = le.fit_transform(labels);

    return X_train_tfidf, Y_labels

target_file_raw = '../data/target.csv'
data_file_raw = '../data/profile_large.json'
data_file_processed = './tmp/data.bigram.temp.npz'
target_file_processed = './tmp/target.bigram.temp.npy'

print "Parsing JSON file..."
contents, targets = prepare_raw_data(target_file_raw, data_file_raw)
X, Y = transform(contents, targets)

if os.path.isfile(data_file_processed):
    os.remove(data_file_processed)
    os.remove(target_file_processed)
save_sparse_csr(data_file_processed, X)
np.save(target_file_processed, Y)
print "File Parsed"
