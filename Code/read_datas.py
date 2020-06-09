def read_data():
    """ Returns training, validation and test data"""
    data_df = pd.read_csv("../Data/imdb_movie_reviews.csv", names=["text", "label"]) #Column 1: any texts, column 2: any binary labels
#    data_df['text'] = data_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    train_df = data_df[1:20001]
    valid_df = data_df[20001:25001]
    test_df = data_df[25001:37501]
    return train_df, valid_df, test_df


def read_data():
    """ Returns training, validation and test data"""
    train_df = pd.read_csv("../Data/dutch_book_reviews_train.csv", names=["text", "label"]) #Column 1: any texts, column 2: any binary labels
    test_df = pd.read_csv("../Data/dutch_book_reviews_test.csv", names=["text", "label"])
    data_df = pd.concat([train_df, test_df[1:]])
    train_df = data_df[1:11867]
    valid_df = data_df[11867:14836]
    test_df = data_df[14836:]
    return train_df, valid_df, test_df


def read_data():
    """ Returns training, validation and test data"""
    text = []
    labels = []
    data_df = pd.read_csv("../Data/five_star_movie_reviews.tsv",sep="\t", names=["PhraseId","SentenceId","Phrase","Sentiment"],low_memory=False) #Column 1: any texts, column 2: any binary labels
    nr_sentences = len(list(set(data_df["SentenceId"].tolist())))
    for i in range(nr_sentences-1):
        try:
            subset = data_df.loc[data_df["SentenceId"]==str((i+1))]
            text.append(subset["Phrase"].tolist()[0])
            labels.append(subset["Sentiment"].tolist()[0])
        except:
            print("Line", str((i+1)), "skipped")
    train_df = pd.DataFrame(list(zip(text[:5000],labels[:5000])), columns=["text", "label"])
#    train_df['text'] = train_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    valid_df = pd.DataFrame(list(zip(text[5000:6000],labels[5000:6000])), columns=["text", "label"])
#    valid_df['text'] = valid_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    test_df = pd.DataFrame(list(zip(text[6000:],labels[6000:])), columns=["text", "label"])
#    test_df['text'] = test_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    return train_df, valid_df, test_df


def read_data():
    """ Returns training, validation and test data"""
    text = []
    labels = []
    with open("../Data/emotion_classification.txt") as file:
        for line in file:
            labels.append(line.split(",")[1][2:-2])
            text.append(line.split(",")[3][2:-5])
    train_df = pd.DataFrame(list(zip(text[1:21334],labels[1:21334])), columns=["text", "label"])
#    train_df['text'] = train_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    valid_df = pd.DataFrame(list(zip(text[21334:26667],labels[21334:26667])), columns=["text", "label"])
#    valid_df['text'] = valid_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    test_df = pd.DataFrame(list(zip(text[26667:],labels[26667:])), columns=["text", "label"])
#    test_df['text'] = test_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    return train_df, valid_df, test_df


def read_data():
    """ Returns training, validation and test data"""
    labels = []
    reviews = []
    with open('../Data/six_class_reviews.txt') as file:
        for line in file:
            labels.append(line.split()[0])
            reviews.append(" ".join(line.split()[3:]))
    data_df = pd.DataFrame(list(zip(reviews, labels)), columns=["text","label"]) #Column 1: any texts, column 2: any binary labels
#    data_df['text'] = data_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    train_df = data_df[:3200]
    valid_df = data_df[3200:4000]
    test_df = data_df[4000:]
    return train_df, valid_df, test_df
