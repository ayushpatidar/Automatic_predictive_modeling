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

    def set(self, data, y, dataset_id, training_id, feature_selector):
        """This function is used to intialize the
        field of the object of algorithms class"""

        self.data_frame = data
        self.target = y
        self.model_name = None
        self.feature_selector = feature_selector
        self.score = None
        self.traning_error = None
        self.dataset_id = dataset_id
        self.training_id = training_id






    def Logistic(self):
        """This function is for Logistic
            Rgeression """
        score = None
        model = None
        error = "{}"
        try:
            model = LogisticRegression(penalty="l1",
                                       dual=False, solver="liblinear")
            results = cross_val_score(model, self.data_frame, self.target, cv=5)
            results = list(results)

            score = np.mean(results)

            print("SCORE IS ", score)

        except Exception as e:
            error = e
            print("error in logistic regression,{}".format(e))

        self.model_name = "LOGISTIC_REGRESSION"
        self.score = score
        self.traning_error = error




    def Decision_tree(self):

        model = None
        score = None
        error = "{}"
        try:
            model = DecisionTreeClassifier(criterion="entropy", splitter="best",
                                           min_samples_split=int(self.data_frame.shape[0]*0.1),
                                           max_features=None)
            results = cross_val_score(model, self.data_frame, self.target, cv=5)
            results = list(results)

            score = np.mean(results)


        except Exception as  e:
            error = e
            print("error in decision_tree classsifying,{}".format(e))


        self.model_name = "DECISION_TREE"
        self.traning_error = error
        self.score = score


    def Random_forest(self):
        """This function implements random
        forest algorithm"""

        model = None
        score = None
        error = "{}"

        try:
            model = RandomForestClassifier(n_estimators=50,
                                           criterion="entropy",
                                           min_samples_split= int(self.data_frame.shape[0]*0.1),
                                           bootstrap=True, oob_score=False,
                                           max_features="sqrt")

            results = list(cross_val_score(model, self.data_frame, self.target, cv=5))
            score = np.mean(results)


        except Exception as e:
            error = e
            print("error in random forest while classifying,{}".format(e))

        self.model_name = "RANDOM_FOREST"
        self.score = score
        self.traning_error = error


    def Knn(self):
        """This function implements knn algorithm"""

        score = None
        model = None
        error = "{}"

        try:
            model = KNeighborsClassifier(n_neighbors=5, weights="uniform", algorithm="auto",
                                 metric="minkowski", n_jobs=-1)

            results = list(cross_val_score(model, self.data_frame, self.target, cv=5))

            score = np.mean(results)


        except Exception as e:
            error = e
            print("error while training in K-nearest neighbours,{}".format(e))

        self.model_name = "K-NEAREST_NEIGHBOURS"
        self.score = score
        self.traning_error = error


    def SGDclassifier(self):

        model = None
        score = None
        error = "{}"

        try:

            model = SGDClassifier(loss="log", penalty="l1", max_iter=2,
                                  learning_rate="optimal")

            results = list(cross_val_score(model, self.data_frame, self.target, cv=3))
            score = np.mean(results)

        except Exception as e:
            error = e
            print("error in training sgd classifier,{}".format(e))

        self.model_name =  "SGD_CLASSIFIER"
        self.score = score
        self.traning_error = error















