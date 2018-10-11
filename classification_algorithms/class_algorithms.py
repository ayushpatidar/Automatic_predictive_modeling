import  os
import  sys
import  warnings
import numpy as np

from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from sklearn.model_selection import cross_val_score


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


    def logistic(self):
        """This function is for Logistic
            Rgeression """
        score = None
        model = None

        try:
            model = LogisticRegression(penalty="l1", dual=False, solver="auto")
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
            mdoel = DecisionTreeClassifier(criterion="entropy", splitter="best",
                                   min_samples_split=df.shape[0]*0.1,
                                   max_features=None,)
            results = cross_val_score(model, self.data_frame, self.target, cv=5)
            results = list(results)

            score = np.mean(results)


        except Exception as  e:
            print("error in decision_tree classsifying,{}".format(e))


        results score