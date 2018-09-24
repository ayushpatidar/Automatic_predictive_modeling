import warnings
from math import sqrt

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import Holt

warnings.filterwarnings("ignore")


def log_transformation(df):
    Y = df.columns.values
    print(df[Y].dtypes)
    print(df.dtypes)
    # MAKE LOG TRASFORMATION
    is_log_transform = False
    """
    the lis will keep track of the indexes which are negative 
    and we have added an constant to make it positive
    
    """
    lis = list()

    try:
        lis = df[Y].values
        min1 = min(lis)

        if min1 <= 0:
            for i in df.index:
                if df[i][Y] < 0:
                    lis.append(i)
                    df[i][Y] = df[i][Y] + abs(min1)

        df_log = np.log(df[Y])
        is_log_transform = True
    except Exception as e:
        print("error in making log transform {}", e)

    return (is_log_transform, df_log)


def train_test_split(df):
    # FUNCTION FOR RETURNING TRAIN AND TEST HOLDOUT SETS 85 PERCENT  IS USED FOR TRAINING AND 15 PERCENT IS USED FOR TESTING
    split = int(df.shape[0] * (0.85))
    train = df[0:split]
    test = df[split:]

    return (train, test)


def TIME_SERIES_ALGO(df, bool_stat):
    dict_rmse = dict()

    bool_log, df_log = log_transformation(df)
    col = df.columns[0]
    # 1.. NAIVE APPROACH
    # IN THIS APPROCAH WE ASSIGN RECENT VALUE TO THE TEST DATAFRAME

    try:
        train, test = train_test_split(df_log)

        y_prd = np.asarray([train.ix[train.shape[0] - 1].values[0]] * (test.shape[0]))

        rs_naive = sqrt(mean_squared_error(test[col].values, y_prd))
        print(rs_naive)
        dict_rmse["naive"] = rs_naive

        if bool_log:
            # PERFORM SAME ABOVE THING FOR LOG TRANSFORMED DATA
            train, test = train_test_split(df_log)

            y_prd = np.asarray([train.ix[train.shape[0] - 1].values[0]] * (test.shape[0]))

            y_prd = np.exp(y_prd)

            rs_naive_log = sqrt(mean_squared_error(test[col].values, y_prd))
            print(rs_naive_log)
            dict_rmse["naive_log"] = rs_naive_log

    except Exception as e:
        print("error in modelling in naive approach,{}".format(e))

    # 2..SIMPLE AVERAGE
    try:

        train, test = train_test_split(df)
        mean_forecast = train[col].mean()
        y_prd = np.asarray([mean_forecast] * test.shape[0])
        rs_mean = sqrt(mean_squared_error(test[col].values, y_prd))
        dict_rmse["simple_avg"] = rs_mean

        if bool_log:
            train, test = train_test_split(df_log)
            mean_forecast = train[col].mean()
            y_prd = np.asarray([mean_forecast] * test.shape[0])

            y_prd = np.exp(y_prd)

            rs_mean = sqrt(mean_squared_error(test[col].values, y_prd))
            dict_rmse["simple_avg_log"] = rs_mean

    except Exception as e:
        print("error in moving average,{}".format(e))

    # 3..MOVING AVERAGE

    # IN PROGRESS HAVE TO MODIFY IT...
    try:
        train, test = train_test_split(df)
        for i in range(25, 90):
            # As rolling mean returns mean fo ecah row we want mean f only last row because it is onlu used to forecast
            mean_moving = train[col].rolling(i).mean().ix[train.shape[0] - 1]
            print(mean_moving)
            y_prd = np.asarray([mean_moving] * test.shape[0])
            rs_moving = sqrt(mean_squared_error(test[col].values, y_prd))
    except Exception as e:
        print("error in moving average,{}".format(e))
    try:

        if bool_log:
            for i in range(25, 90):
                train, test = train_test_split(df_log)

                # print(type(train[col].rolling(i).mean()))
                mean_moving = train[col].rolling(i).mean().ix[train.shape[0] - 1]

                y_prd = np.array([mean_moving] * test.shape[0])
                print(y_prd)
                y_prd = np.exp(y_prd)

                rs_moving_log = sqrt(mean_squared_error(test[col].values, y_prd))

    except Exception as e:
        print("error in log moving average model, {}".format(e))

    # 4.. SIMPLE EXPONENTIAL SMOOTHING
    try:
        train, test = train_test_split(df)
        fit2 = SimpleExpSmoothing(df[col]).fit(smoothing_level=0.6, optimized=False)
        # print(test.index[0])
        # print(test.index[test.shape[0]-1])
        y_prd = fit2.forecast(len(test))
        print(y_prd)

        rs_simple = sqrt(mean_squared_error(test.values, y_prd))
        dict_rmse["simple"] = rs_simple
    except Exception as e:
        print("error is simple exp without log,{}".format(e))

    try:
        if bool_log:
            train, test = train_test_split(df)
            fit2 = SimpleExpSmoothing(df[col]).fit(smoothing_level=0.6, optimized=False)
            y_prd = fit2.forecast(len(test))
            y_prd = np.exp(y_prd)
            rs_simple = sqrt(mean_squared_error(test.values, y_prd))
            dict_rmse["simple_log"] = rs_simple

    except Exception as e:
        print("simple exponential smoothing log,{}".format(e))

    # HOT LINEAR METHOD FOR FORECASTING
    try:
        train, test = train_test_split(df)
        fit2 = Holt(train[col], exponential=True, damped=False).fit()
        y_prd = fit2.predict(test.index.values[0], test.index.values[test.shape[0] - 1])
        rs_hotl = sqrt(mean_squared_error(test[col].values, y_prd))

    except Exception as e:
        print("error in holt linear forecasting in without damped.{}".format(e))

    try:

        fit2 = Holt(train[col], exponential=True, damped=True).fit()
        y_prd = fit2.predict(test.index.values[0], test.index.values[test.shape[0] - 1])
        rs_holtld = sqrt(mean_squared_error(test[col].values, y_prd))

    except Exception as e:
        print("error in holt linear smoothing  damped,{}".format(e))



    #HOLT WINTERS FORECASTING..
