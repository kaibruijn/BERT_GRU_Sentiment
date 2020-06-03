from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn import svm
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.calibration import CalibratedClassifierCV
import math

def preprocessor(x):
    return x

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def print_n_most_informative_features(coefs, features, n):
    # Prints the n most informative features
    most_informative_feature_list = [(coefs[0][nr],feature) for nr, feature in enumerate(features)]
    sorted_most_informative_feature_list = sorted(most_informative_feature_list, key=lambda tup: abs(tup[0]), reverse=True)
    print("\nMOST INFORMATIVE FEATURES\n#\tvalue\tfeature")
    for nr, most_informative_feature in enumerate(sorted_most_informative_feature_list[:n]):
        print(str(nr+1) + ".","\t%.3f\t%s" % (most_informative_feature[0], most_informative_feature[1]))

def read_data():
    transformer_dict = {"negative": -1, "positive": 1}
    df = pd.read_csv("imdb_movie_reviews.csv", names=["text", "labels"])
    train_df = df[1:25001]

    train_df['labels'] = train_df['labels'].apply(lambda x: transformer_dict[x])

    text = []
    labels = []
    df = pd.read_csv("five_star_movie_reviews.tsv",sep="\t", names=["PhraseId","SentenceId","Phrase","Sentiment"],low_memory=False)
    nr_sentences = len(list(set(df["SentenceId"].tolist())))
    for i in range(nr_sentences-1):
        try:
            subset = df.loc[df["SentenceId"]==str((i+1))]
            text.append(subset["Phrase"].tolist()[0])
            labels.append(subset["Sentiment"].tolist()[0])
        except:
            print(str((i+1)), "skipped")

    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['labels'].tolist()
    Xtest = text[6000:]
    Ytest = labels[6000:]
    return Xtrain, Ytrain, Xtest, Ytest


def main():
    Xtrain,Ytrain,Xtest,Ytest = read_data()

    vec = TfidfVectorizer(preprocessor = preprocessor,
                          analyzer = 'word',
                          ngram_range = (1,2))

    classifier = Pipeline( [('vec', vec),
                            ('cls',CalibratedClassifierCV(svm.LinearSVC()))] )

    classifier.fit(Xtrain, Ytrain)

    try:
        coefs = classifier.named_steps['cls'].coef_
        features = classifier.named_steps['vec'].get_feature_names()
        print_n_most_informative_features(coefs, features, 10)
        print()
    except:
        pass

    Yguess = classifier.predict(Xtest)
    probabilities = classifier.predict_proba(Xtest)

    Yguesslist = []
    for guess, proba in zip(Yguess,probabilities):
        if guess == 1:
            pred = proba[1]
        elif guess == -1:
            pred = 1-proba[0]
        Yguesslist.append(pred)

    print(Yguesslist)

    Yguess = Yguesslist
    Ylist = []
    for pred in Yguess:
        pred = pred*5
        pred = round(pred-0.5)
        if pred == 5:
            pred = 4
        Ylist.append(str(int(pred)))


    print(classification_report(Ytest, Ylist))
    print(confusion_matrix(Ytest,Ylist))

    lst = list(zip(Ytest, Ylist))

    large_error = 0
    for i in lst:
        if abs(int(i[0])-int(i[1])) > 1:
            large_error += 1

    print((len(Xtest)-large_error) / len(Xtest))
    print("trained on binary")





if __name__ == '__main__':
    main()
