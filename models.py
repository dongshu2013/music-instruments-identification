from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn import linear_model

class Models():
    mods = {}
    multiclass = True


    def __init__(self, models, multiclass=True):
        self.multiclass = multiclass
        for m in models:
           self.mods[m] = self.create_model(m, multiclass)


    def create_model(self, name, multiclass=True):
        if name == "nb":
            return MultinomialNB()
        elif name == "svm":
            return OneVsRestClassifier(SVC()) if multiclass else SVC()
        elif name == "lr":
            if multiclass:
                return OneVsRestClassifier(linear_model.LogisticRegression(C=1e5))
            else:
                return linear_model.LogisticRegression(C=1e5)
        elif name == "lsvm":
            if multiclass:
                return OneVsRestClassifier(LinearSVC(random_state=0))
            else:
                return LinearSVC(random_state=0)


    def model_names(self):
        return list(self.mods.keys())


    def get_model(self, model_name):
        return self.mods[model_name]


    def is_multiclass():
        return self.multiclass
