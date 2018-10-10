import os
import warnings
import sys
import numpy as np

from sklearn.linear_model import  LogisticRegression
from sklearn.model_selection import cross_val_score


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
        try:
            model = LogisticRegression(penalty="l1", dual=False, solver="auto")
            results = cross_val_score(model, self.data_frame, self.target)
            results = list(results)

            score = np.mean(results)

        except Exception as e:
            print("error in logistic regression,{}".format(e))

        return  score




