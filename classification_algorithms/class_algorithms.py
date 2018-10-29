import os
import  sys
import  warnings
import numpy as np
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier


warnings.filterwarnings('ignore')

class algorithms():
    """all classification algorithms are
    wriiten here"""

    data_frame  = None
    target = None

    def set(self, data, y):
        """This function is used to intialize the
        field of the object of algorithms class"""

        self.data_frame = data
        self.target = y


    def Logistic(self):
        """This function is for Logistic
            Rgeression """
        score = None
        model = None

        try:
            model = LogisticRegression(penalty="l1",
                                       dual=False, solver="auto")
            results = cross_val_score(model, self.data_frame, self.target, cv=5)
            results = list(results)

            score = np.mean(results)

        except Exception as e:
            print("error in logistic regression,{}".format(e))

        return (score, model)


    def Decision_tree(self):

        model = None
        score = None

        try:
            model = DecisionTreeClassifier(criterion="entropy", splitter="best",
                                           min_samples_split=self.data_frame.shape[0]*0.1,
                                           max_features=None,)
            results = cross_val_score(model, self.data_frame, self.target, cv=5)
            results = list(results)

            score = np.mean(results)


        except Exception as  e:
            print("error in decision_tree classsifying,{}".format(e))


        return  (score, model)


    def Random_forest(self):
        """This function implements random
        forest algorithm"""

        model = None
        score = None

        try:
            model = RandomForestClassifier(n_estimators=50,
                                           criterion="entropy",
                                           min_samples_split=self.data_frame.shape[0]*0.1,
                                           bootstrap=True, oob_score=False,
                                           max_features="sqrt")

            results = list(cross_val_score(model, self.data_frame, self.target, cv=5))
            score = np.mean(results)


        except Exception as e:
            print("error in random forest while classifying,{}".format(e))

        return (score, model)


    def Knn(self):
        """This function implements knn algorithm"""

        score = None
        model = None

        try:
            model = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto",
                                 metric="minkowski", n_jobs=-1)

            results = list(cross_val_score(model, self.data_frame, self.target, cv=5))

            score = np.mean(results)


        except Exception as e:
            print("error while training in K-nearest neighbours,{}".format(e))

        return (score, model)

    def SGDclassifier(self):

        model = None
        score = None

        try:

            model = SGDClassifier(loss="log", penalty="l1", max_iter=2,
                                  learning_rate="oprimal")

            results = list(cross_val_score(model, self.data_frame, self.target, cv=3))
            score = np.mean(results)

        except Exception as e:
            print("error in training sgd classifier,{}".format(e))

        return  (score, model)















