import os
import pickle
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


warnings.filterwarnings('ignore')


def Decision_tree(Y):
    path = "pickle_dumps"

    for fs in os.listdir(path):
        print(fs)
        if fs == "__init__.py":
            continue
        df = open(path+"/"+str(fs),"r")
            #print(df)
        print(df)
        df = pickle.load(df)
        print(type(df))

        res_dc = list()

        try:
            dc = DecisionTreeClassifier(criterion = "entropy",splitter = "best",min_samples_split = 0.08,max_features = None)
            #WRITE ALL ALGORITHMS HERE ONLY....
            res_dc = list(cross_val_score(dc,df,Y,cv = 5))
            print(res_dc)
            return list(res_dc)

        except Exception as e:
            print("****error while classifying int decision tree****",e)






def class_algo(Y):

    print("*********in classification algorithms*********")

    #1.DECISION  TREE
    try:
        print("decision classifier is called")
        Decision_tree(Y)
        print("*******function returened from decision classifier******")


    except Exception as e:
        print("error in decision tree classifier while calling ",e)


    #2.LOGISTIC REGRESSION

