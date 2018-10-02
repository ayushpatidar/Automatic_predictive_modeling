import argparse
import warnings

import pandas as pd
import pandas as np

from null_values.null_values_treatment import null_treatment

from outliears_values.outliears_values_treatment import outliears_treatment
from spaces.feature_selection_spaces import features_spaces
from feature_selection.classification_feature_selection import feature_classification
from feature_encoding.feature_encoding_file import feature_encoding
from classification_algorithms.class_algorithms import  class_algo

warnings.filterwarnings('ignore')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="modelling")
    parser.add_argument('--dataframe', type=str, help="reading dataframe")
    parser.add_argument("--target", type=str, help="load target variable")
    parser.add_argument("--type", type=str, help="problem type")

    args = parser.parse_args()

    if args.dataframe is None:
        print("****dataframe is not given****")
        exit()

    if args.target is None:
        print("****target variable is not given****")
        exit()

    # reading datafarme
    try:
        print("*****loading dataset******")
        df = pd.read_csv(args.dataframe + ".csv")


        if args.target not in df.columns:
            print("Specified target variable is not in the dataframe")
            exit()



        print(df.head())
        print("*****dataset completely loaded******")

        X = df.drop(args.target, axis=1)
        y = df[args.target]
        print(y)

        if args.type is None:
            if len(y.unique()) < 0.25 * len(y):
                type = "classification"
            else:
                type = "regression"

        else:
            type = args.type

        print("*****calling null values funcion******")
        # print("***total null values is whole dataframe",df.isnull().sum())
        df = null_treatment(df)
        print("*****function returned from null values******")

        print("********calling featutre encoding function*********")

        df = feature_encoding(df)

        print("******function returned from label encoder**********")

        print("*****calling outliers function******")
        X, y = outliears_treatment(X, y)

        print("******function retured from outliers treatment******")

        print("******feature selection started******")

        print("****call for feature space is made******")

        dict_spaces = features_spaces(type)

        print("*****funtion retured form feature spaces***")

        if (type == "classification"):
            print("***feature selection for classification is chosen*****")

            feature_classification(X, y)

            print("****function returned from feature selection for classifiction****")




        print(type)       #prints type of dataset it is...


        print("calling classification algo")
        class_algo(y)


    except Exception as e:
        print("error", e)
