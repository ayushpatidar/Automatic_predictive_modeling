from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')


def feature_encoding(df):
    try:

        le = LabelEncoder()
        for i in df.columns:
            if(df[i].dtype=="object"):
                df[i] = le.fit_transform(df[i])




    except Exception as e:
        print("error in feature encoding", e)
