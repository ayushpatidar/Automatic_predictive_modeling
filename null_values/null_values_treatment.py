import warnings
import numpy as np

warnings.filterwarnings('ignore')


def null_treatment(df):
    print("*****null values treatment started*******")

    cols = df.columns  # stores columns
    total = 0  # stores number of null values in a columns
    rows = df.shape[0]  # stores number of rows in  a dataframe
    null_lis = list()  # stores column that have to be deleted if threshold of null values increases

    for i in cols:

        total = df[i].isnull().sum()  # storing null values present in colums

        if total / rows > 0.45:   #theshold for null value to drop a columns is 45 percent
            null_lis.append(i)  # append into list so that we can drop column

        else:
            if df[i].dtype in [np.number]:
                mean_val = df[i].mean(axis=0)
                # it means column is numeric
                print("*****column is numeric******")
                # preprocessing.Imputer(missing_values="Nan", strategy='mean').fit_transform(df[i].values)
                df[i].fillna(value=mean_val, inplace=True)





            else:
                # it menas column is not numeric
                print("******column is of object type******")
                mode_val = df[i].mode()
                # preprocessing.Imputer(missing_values="Nan",strategy='most_frequent').fit_transform(df[i].values)
                df[i].fillna(value=mode_val, inplace=True)

    print("*****null values treatment finished******")

    return (df)  # returning transformed dataframe
