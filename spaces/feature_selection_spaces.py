import warnings

warnings.filterwarnings("ignore")


def features_spaces(type):
    dic_class = dict()
    dic_reg = dict()

    # a features selection space which will retuen whole spaces for features selection for a given type of algorithm
    if type == "classification":
        print("******feature space of classification is called*******")

        dic_class={"1": "pca", "2": "lda", "3": "slectkbest",
               "4": "selectfpr", "5": "selectfdr"}

        print("*****feature space from classification is retured********")

        return  dic_class




    else:
        print("******feature space of regression is called********")


