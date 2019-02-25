import pickle
import warnings

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
warnings.filterwarnings('ignore')

space_score_func = list()

space_score_func = [f_classif, mutual_info_classif, chi2]


def feature_classification(df, Y):
    try:  # DIMENSIONALITY REDUCTION AND FEATURE SELECTION

        scaler = StandardScaler()  # applying standared scaler because  pca works best for normaliz data
        f = open("pickle_dumps/pca_feature_dumps", "wb")  # opening file in which we want to dump
        print("***appying pca****")
        pca_df = PCA(n_components=0.85, svd_solver="auto").fit_transform(
            scaler.fit_transform(df))  # applying pca n components specify how much varience retention we want
        print(pca_df.shape)
        print(df.shape)
        pickle.dump(pca_df, f)  # dumping pca results into file
        f.close()

        print("****PCA finished*****")

    except Exception as e:
        print("error in pca section of feature selection", e)




    try:  # RECURSIVE FEATURE SELECTION
        print("*****recursive feature selection****")  # rfe selects feature recursively
        # and eleminates features as it's import goes less
        s = SVC(kernel="linear")
        selec = RFECV(estimator=s, step=1, cv=3, scoring=None)
        tmp = selec.fit_transform(df, Y)
        f = open("pickle_dumps/rfe_feature_dumps", "wb")
        pickle.dump(tmp, f)
        f.close()
        print("******rfe finished***********")

    except Exception as e:
        print("error in rfe feature selection", e)

    try:  # TREE BASED FEATURE SELECCTION
        print("*****selectfrommodel in feature selection*******")
        rf = RandomForestClassifier()
        rf = rf.fit(df, Y)
        tmp = SelectFromModel(rf, threshold=None, prefit=True).transform(df)
        f = open("pickle_dumps/selcetfrommodel_feature_selection", "wb")
        pickle.dump(tmp, f)
        f.close()

        print("********select_from_model finsihed ********")

    except Exception as e:
        print("error", e)



    try:  # UNIVARIATE FEATURE SELECTION
        print("****selctkbest is started for feature selection***")
        for i in space_score_func:
            print(i)
            tmp = SelectKBest(i, int(0.6 * df.shape[1])).fit_transform(df, Y)
            f = open("pickle_dumps/feature_selectkbest"+str(i), "wb")
            pickle.dump(tmp, f)
            f.close()

        print("*********selctkbest has finished*********")



    except Exception as e:
        print("error occured in selectkbest", e)


    print("********selctfpr for feature selection is started**********")

    for i in space_score_func:
        try:
            print(i)
            tmp = SelectFpr(i, alpha=0.05).fit_transform(df, Y)

            f = open("pickle_dumps/selectfpr_feature_selection"+str(i), "wb")
            pickle.dump(tmp, f)
            f.close()

        except Exception as e:
            print("error in feature selection in selectfpr",e)

    print("********selctfpr for feature selection is finished")





    print("********feature selection in selectfdr****")
    for i in space_score_func:
        try:

            tmp = SelectFdr(i, alpha=0.05).fit_transform(df, Y)
            f = open("pickle_dumps/selectfdr_fs"+str(i), "wb")
            pickle.dump(tmp, f)
            f.close()

        except Exception as e:
            print("error in feature selection in selectfdr",e)

    print("**********selectfdr in feature selection finished**************")



