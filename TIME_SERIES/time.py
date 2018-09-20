import argparse
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from TIME_SERIES.stationary.stationary_test import test_stationary

if __name__ == "main":

    parser = argparse.ArgumentParser(description="tims_series")

    parser.add_argument("--dataframe", type=str, help="dataframe")
    parser.add_argument("--target", type=str, help="target")

    args = parser.parse_args()

    if args.dataframe == None:
        print("dataframe not given")
        exit()
    if arge.target == None:
        print("target is not given")
        exit()

    df = args.dataframe
    target = args.target

    #filling null values with the forward fiiling method
    df = df.fillna(method = "ffill")


    df = pd.read_csv(str(dataframe) + ".csv")
    col = df.columns
    for i in col:
        try:
            if i != target:
                # converting column into date time format
                df[i] = pd.to_datetime(df[i])

                # setting index of dataframe as date time column
                df.index = df[i]

                # removing that column because now it is not needed
                df.drop(i, axis=1, inplace=True)

        except Exception as e:
            print("error while parsing into date error{}", format(e))

    bool = test_stationary(df, target)

    if bool == True:
# need to make series stationary
















