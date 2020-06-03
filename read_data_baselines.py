def read_data(): #English binary sentiment analysis task
    df = pd.read_csv("imdb_movie_reviews.csv", names=["text", "labels"])
    train_df = df[1:25001]
    test_df = df[25001:37501]
    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['labels'].tolist()
    Xtest = test_df['text'].tolist()
    Ytest = test_df['labels'].tolist()
    return Xtrain, Ytrain, Xtest, Ytest


def read_data(): #Dutch binary sentiment analysis
    train_df = pd.read_csv("dutch_movie_reviews_train.csv", names=["text", "labels"])
    test_df = pd.read_csv("dutch_movie_reviews_test.csv", names=["text", "labels"])
    df = pd.concat([train_df, test_df[1:]])
    train_df = df[1:14836]
    test_df = df[14836:]
    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['labels'].tolist()
    Xtest = test_df['text'].tolist()
    Ytest = test_df['labels'].tolist()
    return Xtrain, Ytrain, Xtest, Ytest


def read_data(): #5 star multiclass sentiment analysis
    transformer_dict = {"negative": -1, "positive": 1}


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

    Xtrain = text[:6000]
    Ytrain = labels[:6000]
    Xtest = text[6000:]
    Ytest = labels[6000:]
    return Xtrain, Ytrain, Xtest, Ytest


def read_data(): #Emotion multiclass sentiment analysis
    text = []
    labels = []
    with open("text_emotion.txt") as file:
        for line in file:
            labels.append(line.split(",")[1][2:-2])
            text.append(line.split(",")[3][2:-5])
    train_df = pd.DataFrame(list(zip(text[1:26667],labels[1:26667])), columns=["text", "label"])
    test_df = pd.DataFrame(list(zip(text[26667:],labels[26667:])), columns=["text", "label"])
    Xtrain = train_df['content'].tolist()
    Ytrain = train_df['sentiment'].tolist()
    Xtest = test_df['content'].tolist()
    Ytest = test_df['sentiment'].tolist()
    return Xtrain, Ytrain, Xtest, Ytest




def read_data(): #lfd multiclass task
    """ Returns training, validation and test data"""
    labels = []
    reviews = []
    with open('six_class_reviews.txt') as file:
        for line in file:
            labels.append(line.split()[0])
            reviews.append(" ".join(line.split()[3:]))
    data_df = pd.DataFrame(list(zip(reviews, labels)), columns=["text","label"]) #Column 1: any texts, column 2: any binary labels
#    data_df['text'] = data_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    train_df = data_df[:4000]
    test_df = data_df[4000:]
    Xtrain = train_df['text'].tolist()
    Ytrain = train_df['label'].tolist()
    Xtest = test_df['text'].tolist()
    Ytest = test_df['label'].tolist()
    return Xtrain, Ytrain, Xtest, Ytest
