import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import Imputer


def results(df):
    """
    0th column is discarded,First column is taken as input regressor/classifer,rest are taken as input.Last one is time
    Arguments:
    df -- dataframe with Regressor name at front and time at back
    Returns:
    2 Dicts, sum and time of each regressor are returned back
    """
    Sum = {}
    time = {}
    n = len(df.columns)
    for i in df.index:
        Sum[df.iloc[i, 1] + ' Sum'] = df.iloc[i, 2:n - 1].sum()
        time[df.iloc[i, 1] + ' Time'] = df.iloc[i, n - 1]
    return Sum, time


class MiceImputer:
    def __init__(self, seed_nulls=False, seed_strategy='mean'):
        self.seed_nulls = seed_nulls
        self.seed_strategy = seed_strategy
        mdict = {}
        self.model_dict_ = mdict

    def get_model_dict(self):
        return self.model_dict_

    def transform(self, X):
        if type(X) == np.ndarray:
            X = pd.DataFrame(X)
        col_order = X.columns
        new_X = []
        mutate_cols = list(self.model_dict_.keys())
        #      print(mutate_cols)
        for i in mutate_cols:
            y = X[i]
            x_null = X
            y_null = y[y.isnull()].reset_index()['index']

            y_notnull = y[y.notnull()]

            imp = Imputer(strategy=self.seed_strategy)
            model = self.model_dict_.get(i)
            if self.seed_nulls:
                x_null = pd.DataFrame(model[1].transform(x_null))
            else:
                x_null1 = x_null
                null_check = x_null.isnull().any()
                x_null = x_null[null_check.index[~null_check]]

            if x_null.shape[1] ==0 and not self.seed_nulls:
                x_null = pd.DataFrame(imp.fit_transform(x_null1))
                x_null.drop(i,axis = 1,inplace = True)



            x_null = x_null[y.isnull()]
           # print(x_null.shape)
           # pwd = model[0].predict(x_null)



            pred = pd.concat([pd.Series(model[0].predict(x_null)) \
                             .to_frame() \
                             .set_index(y_null), y_notnull], axis=0) \
                .rename(columns={0: i})
            new_X.append(pred)

        new_X.append(X[X.columns.difference(mutate_cols)])

        final = pd.concat(new_X, axis=1)[col_order]

        return final

    def fit(self, X, model=None, max_cat=12):

        if type(X) == np.ndarray:
            X = pd.DataFrame(X)
        x = X.fillna(value=np.nan)

        null_check = x.isnull().any()  # true aur false to column
        null_data = x[null_check.index[null_check]]  # chooses columns with null data

        for i in null_data:
            y = null_data[i]  # null_data
            y_notnull = y[y.notnull()]

            model_list = []  #
            if self.seed_nulls:
                imp = Imputer(strategy=self.seed_strategy)
                model_list.append(imp.fit(x))
                non_null_data = pd.DataFrame(imp.fit_transform(x))

                non_null_data = non_null_data.drop(i, axis=1)
            else:
                non_null_data = x[null_check.index[~null_check]]  # takes data that is not null



            if non_null_data.shape[1] == 0 and not self.seed_nulls:
                imp = Imputer(strategy=self.seed_strategy)
                model_list.append(imp.fit(x))
                non_null_data = pd.DataFrame(imp.fit_transform(x))

                non_null_data = non_null_data.drop(i, axis=1)

            x_notnull = non_null_data[y.notnull()]

            #    print(model)
            if model == None:
                if y.nunique() > max_cat:  #
                    model = LinearRegression()
                else:
                    model = LogisticRegression()
            # print('x_dtypes',x_notnull.dtypes,'y_notnull_types',y_notnull.dtypes)
            # print(x_notnull.shape,'\nY',y_notnull.shape,'\n')

            model.fit(x_notnull.values, y_notnull.values)
            model_list.insert(0, model)
            #print(model_list)
            self.model_dict_.update({i: model_list})

        return self

    def fit_transform(self, X, model=None, max_cat=12):

        """
        fits the model for data imputation
        Arguments:
        X -- input dataframe to be iputed
        model -- The type of model needed
        max_cat -- tells the number of category if we want for categorical data
        Returns:
        Data -- imputed data
        """
        return self.fit(X, model, max_cat).transform(X)


if __name__ == '__main__':
    import numpy as np
    X = np.random.randn(1000, 3)

    import pandas as pd

    X  = pd.DataFrame(X)

    X.iloc[0, 0] = None
    X.iloc[2, 1] = None
    X.iloc[0, 2] = None

    print(X)
    m = MiceImputer(seed_nulls=False)
    X = m.fit_transform(X)

    print(X)