from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import linear_model
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer


def preprocessor(x):
    return x

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))

def read_data(): #Dutch binary sentiment analysis
    train_df = pd.read_csv("../Data/dutch_book_reviews_train.csv", names=["text", "labels"])[1:]
    test_df = pd.read_csv("../Data/dutch_book_reviews_test.csv", names=["text", "labels"])[1:]
    df = pd.concat([train_df, test_df[1:]])
    print(df)
    transformer_dict = {"0": -10, "1": 10}
    df['labels'] = df['labels'].apply(lambda x: transformer_dict[x])
    train_df = df[1:14836]
    test_df = df[14836:]
    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['labels'].tolist()
    Xtest = test_df['text'].tolist()
    Ytest = test_df['labels'].tolist()
    return Xtrain, Ytrain, Xtest, Ytest


def main():
    Xtrain,Ytrain,Xtest,Ytest = read_data()

    vec = TfidfVectorizer(preprocessor = preprocessor)

    classifier = Pipeline( [('vec', vec),
                            ('cls',linear_model.LinearRegression())] )

    classifier.fit(Xtrain, Ytrain)

    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass
    Ylist = []
    Ytestlist = []
    Yguess = classifier.predict(Xtest)
    for guess in Yguess:
        if guess < 0:
            Ylist.append('negative')
        else:
            Ylist.append('positive')
    for Ytrue in Ytest:
        if Ytrue < 0:
            Ytestlist.append('negative')
        else:
            Ytestlist.append('positive')

    print(classification_report(Ytestlist, Ylist))
    print(confusion_matrix(Ytestlist,Ylist))


if __name__ == '__main__':
    main()
