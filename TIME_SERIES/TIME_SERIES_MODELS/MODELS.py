from math import sqrt
import numpy as np
from sklearn.metrics import mean_squared_error


def log_transformation(df):
    #MAKE LOG TRASFORMATION
    is_log_transform = False

    try:
        lis = df[Y]
        min1 = min(lis)
        if min1 <= 0:
            for i in df.index:
                if df[i][Y] < 0:
                    df[i][Y] = df[i][Y] + abs(min1)

        df_log = log(df)
        is_log_transform = True
    except Exception as e:
        print("error in making log transform {}", e)

    return  (is_log_transform,df_log)



def train_test_split(df):

    #FUNCTION FOR RETURNING TRAIN AND TEST HOLDOUT SETS 85 PERCENT  IS USED FOR TRAINING AND 15 PERCENT IS USED FOR TESTING
    split = int(df.shape[0]*(0.85))
    train = df[0:split]
    test = df[split:]

    return (train,test)



def TIME_SERIES_ALGO(df,bool_stat):

    bool_log,df_log = log_transformation()
    col = df.columns[0]
    #1.. NAIVE APPROACH
    #IN THIS APPROCAH WE ASSIGN RECENT VALUE TO THE TEST DATAFRAME

    try:
        train, test = train_test_split(df_log)

        y_prd = np.asarray([train.ix[train.shape[0] - 1].values[0]] * (test.shape[0]))

        rs_naive = sqrt(mean_squared_error(test[col].values, y_prd))
        print(rs_naive)

        if bool_log:
            #PERFORM SAME ABOVE THING FOR LOG TRANSFORMED DATA
            train,test = train_test_split(df_log)

            y_prd = np.asarray([train.ix[train.shape[0]-1].values[0]]*(test.shape[0]))

            y_prd = np.exp(y_prd)

            rs_naive_log = sqrt(mean_squared_error(test[col].values,y_prd))
            print(rs_naive_log)

    except Exception as e:
        print("error in modelling in naive approach,{}".format(e))


    #2..SIMPLE AVERAGE
    try:

        train,test = train_test_split(df)
        mean_forecast = train[col].mean()
        y_prd = np.asarray([mean_forecast]*test.shape[0])

        rs_mean = sqrt(mean_squared_error(test[col].values,y_prd))

        if bool_log:
            train, test = train_test_split(df_log)
            mean_forecast = train[col].mean()
            y_prd = np.asarray([mean_forecast] * test.shape[0])

            y_prd = np.exp(y_prd)

            rs_mean = sqrt(mean_squared_error(test[col].values, y_prd))

    except Exception as e:
        print("error in moving average,{}".format(e))


    #3..MOVING AVERAGE
    try:
        train,test = train_test_split(df)
        for i in range(25,90):
            mean_moving = train[col].rolling(i).mean()
            y_prd = np.asarray([mean_moving]*test.shape[0])
            rs_moving = sqrt(mean_squared_error(test[col].valus,y_prd))

            if bool_log:
                train,test = train_test_split(df_log)
                mean_moving = train[col].rolling(i).mean()
                y_prd = np.asarray([mean_moving]*test.shape[0])
                y_prd = np.exp(y_prd)
                rs_moving_log = sqrt(mean_squared_error(test[col].values,y_prd))

    except Exception as e:
        print("error if moving average model, {}".format(e))




































