import argparse
import warnings

import pandas as pd

#from null_values.null_values_treatment import null_treatment

warnings.filterwarnings('ignore')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="modelling")
    parser.add_argument('--dataframe', type=str, help="reading dataframe")
    parser.add_argument("--target", type=str, help="load target variable")

    args = parser.parse_args()

    if args.dataframe is None:
        print("dataframe is not given")
        exit()

    if args.target is None:
        print("target variable is not given")
        exit()

        # reading datafarme
    try:
        print("loading dataset")
        df = pd.read_csv(args.dataframe + ".csv")

        col = df.columns
        if args.target not in col:
            print("error")
            exit()
        print(df.head())
        print(df.index)






    except Exception as e:
        print("error ",e)




          #chi2 test


