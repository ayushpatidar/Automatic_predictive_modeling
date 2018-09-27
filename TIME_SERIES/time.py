import argparse
import warnings

warnings.filterwarnings('ignore')
import pandas as pd
from stationary.stationary_test import test_stationary
from TIME_SERIES_MODELS.MODELS import TIME_SERIES_ALGO

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="tims_series")

    parser.add_argument("--dataframe", type=str, help="dataframe")
    parser.add_argument("--target", type=str, help="target")

    args = parser.parse_args()
    if args.dataframe == None:
        print("dataframe not given")
        exit()
    if args.target == None:
        print("target is not given")
        exit()

    df = args.dataframe
    target = args.target


    #filling null values with the forward fiiling method
    #df = df.fillna(method = "ffill")

    path = "TIME_SERIES"+"/"
    df = pd.read_csv(path+str(df) + ".csv")
    col = df.columns

    if target not in col:
        print("given target variable is not in the dataframe")
        exit()



    print(df.head())
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
    print("dataframe after parsing")
    #print(df.head())


    bool = test_stationary(df, target)
    print(bool)

    TIME_SERIES_ALGO(df,bool)


# need to make series stationary
















