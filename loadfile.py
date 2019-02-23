import argparse
import warnings

import pandas as pd
import pandas as np

from null_values.null_values_treatment import null_treatment

from outliears_values.outliears_values_treatment import outliears_treatment
from spaces.feature_selection_spaces import features_spaces
from feature_selection.classification_feature_selection import feature_classification
from feature_encoding.feature_encoding_file import feature_encoding
from classification_algorithms.class_algorithms import algorithms
from mysqlclient import commit_results_db
from mysqlclient import set_results_db
from mysqlclient import fetch_results
import os
import pickle
import uuid
import hashlib

warnings.filterwarnings('ignore')


#USE IT FOR COMMAND LINE EXECUTION

# if __name__ == '__main__':
#
#     parser = argparse.ArgumentParser(description="modelling")
#     parser.add_argument('--dataframe', type=str, help="reading dataframe")
#     parser.add_argument("--target", type=str, help="load target variable")
#     parser.add_argument("--type", type=str, help="problem type")
#
#     args = parser.parse_args()
#
#     if args.dataframe is None:
#         print("****dataframe is not given****")
#         exit()
#
#     if args.target is None:
#         print("****target variable is not given****")
#         exit()
#


def main_function(file_path, target_name, type):

    args = dict()
    args["dataframe"] = file_path
    args["target"] = target_name
    args["type"] = type

    #unique dataset id
    dataset_id= str(hashlib.sha1())


    df = None

    #reading dataframe
    try:
        print("--------------------------")
        print("*****loading dataset******")
        df = pd.read_csv(args["dataframe"] + ".csv")


        if args["target"] not in df.columns:
            print("Specified target variable is not in the dataframe")
            exit()



        print(df.head())
        print("*****dataset completely loaded******")


        print("-------------------------------")
        print(" ")
        X = df.drop(args["target"], axis=1)
        y = df[args["target"]]
        print(y)

        if args["type"] is None:
            if len(y.unique()) < 0.25 * len(y):
                type = "classification"
            else:
                type = "regression"

        else:
            type = args["type"]






        print("---------------------------------")
        print(" ")

        print("*****calling null values funcion******")
        # print("***total null values is whole dataframe",df.isnull().sum())
        df = null_treatment(df)
        print("*****function returned from null values******")

        print("---------------------------------")
        print(" ")




        print("********calling featutre encoding function*********")

        df = feature_encoding(df)

        print("******function returned from label encoder**********")

        print("---------------------------------")
        print(" ")





        print("*****calling outliers function******")
        X, y = outliears_treatment(X, y)

        print("******function retured from outliers treatment******")

        print("---------------------------------")
        print(" ")






        print("******feature selection started******")


        if (type == "classification"):
            print("***feature selection for classification is chosen*****")

            feature_classification(X, y)

            print("****function returned from feature selection for classifiction****")

            print("---------------------------------")
            print(" ")







        print(type)       #prints type of dataset it is...

        print("---------------------------------")
        print(" ")

        print("calling classification algo")


        dump_path = "/home/ayushpatidar/PycharmProjects/Automatic_predictive_modeling/pickle_dumps"

        for feature_selector in os.listdir(dump_path):

            f = open(dump_path + "/" +str(feature_selector), "rb")
            feature_selector_current = pickle.load(f)
            print("feature_selector is", feature_selector)


            #creating object of class algorithms
            obj = algorithms()

            training_id = str(uuid.uuid4())

            X  = feature_selector_current
            obj.set(X, y, dataset_id, training_id, str(feature_selector))
            obj.Logistic()


            commit_results_db()
            data = (obj.dataset_id, obj.training_id, obj.model_name,
                           obj.score, obj.feature_selector, obj.traning_error)
            set_results_db(data)


            training_id = str(uuid.uuid4())
            obj.training_id = training_id
            obj.Decision_tree()
            commit_results_db()
            data = (obj.dataset_id, obj.training_id, obj.model_name,
                    obj.score, obj.feature_selector, obj.traning_error)
            set_results_db(data)



            training_id = str(uuid.uuid4())
            obj.training_id = training_id
            obj.Random_forest()
            commit_results_db()

            data = (obj.dataset_id, obj.training_id, obj.model_name,
                    obj.score, obj.feature_selector, obj.traning_error)
            set_results_db(data)



            training_id = str(uuid.uuid4())
            obj.training_id = training_id
            obj.SGDclassifier()
            commit_results_db()
            data = (obj.dataset_id, obj.training_id, obj.model_name,
                    obj.score, obj.feature_selector, obj.traning_error)
            set_results_db(data)


        print("ALL ALGORITHMS ARE COMPLETED")


        df = fetch_results()
        df = df.sort_values(by="ACCURACY", ascending=False).head(10)

        print(df)

    except Exception as e:
        print("error", e)


    return df


