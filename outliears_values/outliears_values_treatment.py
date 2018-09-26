import warnings
from math import ceil

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest

warnings.filterwarnings('ignore')


def voting(db_ans,iso_ans):          #voting funtion which combine result of dbscan and isolation forest

    try:
        ans = list()
        for i in range(len(db_ans)):
            if db_ans[i]==1 or iso_ans[i]==1:  #appending those rows which are not outliers
                ans.append(i)

        return  ans              #return the list containg index of rows which are not outliers

    except Exception as e:
        print("error in voting",e)


def calculate_eps(k, df):  # function for calculating eps for DBSCAN

    try:
        #print(k,df.head())
        nbrs = NearestNeighbors(n_neighbors = k, algorithm="auto").fit(df)  # finding n neighboues
        distances, indices = nbrs.kneighbors(df)  # distances contains distance of all it's k neighbours

        shap = distances.shape[0]  # shap stores shape of distances nd- array

        lis_mean_n_distances = list()  # stores the mean distance of every row of distances nd -array
        for i in range(shap):
            x = np.mean(distances[i])  # calculating mean of every row of distances nd array
            lis_mean_n_distances.append(x + x * 0.2)  # setting means a bit higher than mean 20 percent

        ans = np.mean(lis_mean_n_distances)  # ans has the mean of all the rows of nd-array

    except Exception as e:
        print("error in calculating eps DBSCAN ",e)

    return ans + 0.1 * ans  # adding 10 percent threshold to the ans value


def outliears_treatment(df,y):
    print("******outliers started******")

    eps_cal = calculate_eps(int(ceil(0.08 * df.shape[0])),df)  # function to calculate the eps distance for DBSCAN  minimum point
    # nearset neighbour are taken 0.8 percent generally taken sqrt(n)/2
    print("******execution of dbscan started*******")
    db_ans= DBSCAN(eps=eps_cal, min_samples=0.05 * df.shape[0]).fit_predict(df)     #predicting outliers values using dbscan

    print("******dbscan finidshed******")


    print("******isolationforest method for outliers setection******")

    iso = IsolationForest(n_estimators=100,max_samples='auto')
    iso.fit(df)
    iso_ans = iso.predict(df)



    print("******isolation forest execution completed*******")

    print("****apply voting for both of them******")
    final_ans=voting(db_ans,iso_ans)
    print("*********voting funcion for outliers finished********")


    return (df.ix[final_ans],y.ix[final_ans])    #returning tranformed dataframe after oulier detection















