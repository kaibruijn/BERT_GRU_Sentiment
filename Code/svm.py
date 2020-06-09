from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

def preprocessor(x):
    return x

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))

def read_data(): #Copy appropriate read_data function from read_data_baselines.py
    print("Copy appropriate read_data function from read_data_baselines.py")

def main():
    Xtrain,Ytrain,Xtest,Ytest = read_data()

    vec = TfidfVectorizer(preprocessor = preprocessor,
                          analyzer = 'word',
                          ngram_range = (1,2))

    classifier = Pipeline( [('vec', vec),
                            ('cls',svm.LinearSVC())] )

    classifier.fit(Xtrain, Ytrain)

    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass

    Yguess = classifier.predict(Xtest)

    print(classification_report(Ytest, Yguess))
    print(confusion_matrix(Ytest,Yguess))


if __name__ == '__main__':
    main()
