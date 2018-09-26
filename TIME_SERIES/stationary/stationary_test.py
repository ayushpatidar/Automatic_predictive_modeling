import warnings
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

def test_stationary(df,Y):
    #function to check whether datafrmae is stationay or not

    #METHOD-1 (COMPARING MEAN AND VARIENCE AT DIFFRENT TIMES)
    s = 0
    ns = 0
    try:

        val = df[Y].values

        #spliting the valuses into two sets
        split = int(len(val)/2)

        left = val[0:split]
        right = val[split:]

        #calcualting the mean and varience of the splitted list
        mean1 = left.mean()
        mean2 = right.mean()

        var1 = left.var()
        var2 = right.var()

        s = 0
        ns = 0
        if abs(mean1-mean2)<=10 and abs(var1-var2)<=10:
            #it means time series is stationary
            s = 1
        else:
            ns = ns+1


    except Exception as e:
        print("error in method 1 of stationay {}",format(e))


    #METHOD-2 (DICKEY-FULLER TEST)
    try:
        res = adfuller(df[Y],autolag = "AIC")
        if res[1]>=0.05:
            #series is not statioanry
            s = s+1
            #return  True
        else:
            #series is stationay
            ns = ns+1
            #return  False

    except Exception as e:
        print("error in performing dicky fuller test error{}",format(e))


    if s>1:
        return  True
    else:
        if ns>1:
            return  False
        else:
            if s==1:
                return  True
            else:
                return  False



