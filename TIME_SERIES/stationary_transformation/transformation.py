import pandas as pd
import numpy as np
import pickle


def transformation_stationary(df, Y):
    #1.. MAKING LOG TRANSFORM to reduce trend
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

    #2.. MOVING AVERAGE

    try:
        if is_log_transform:
            # MEANS APPLY TRANSFORMATION ON LOG TRANSFORMED DATAFRAME

            df_rolling = df_log.rolling(window = 12).mean()
            df_rolling = df_log - df_rolling

        else:
            # APPLY TRANSFORMATION ON ORIGINAL DATAFARME

            df_rolling = df.rolling(window = 12).mean()
            df_rolling = df - df_rolling

        f = open("TIME_SERIES/pickle_dumps/TIME_SERIES")
        pickle.dump(df_rolling, f)
        f.close()

    except Exception as e:
        print("error in moving avergae in transformation,{}".format(e))


    #3..FIRST-DIFFRENCING THE DATAFRAME
    try:




    except Exception as e:
        print("error in diffrencing in transformation module,{}".format(e))


    #4..SEASONAL DIFFRENCING

    try:


    except Exception as e:
        print("error in seasonal diffrencing,{}".format(e))
