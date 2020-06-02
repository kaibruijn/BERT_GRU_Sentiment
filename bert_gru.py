from langdetect import detect
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from torchtext.data import Field, Dataset, Example, BucketIterator
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import time
import torch


class DataFrameDataset(Dataset): # Thanks to https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext
    """Class for using pandas DataFrames as a datasource""" 
    def __init__(self, examples, fields, filter_pred=None):
        """
        Create a dataset from a pandas dataframe of examples and Fields
        Arguments:
            examples pd.DataFrame: DataFrame of examples
            fields {str: Field}: The Fields to use in this tuple. The
                string is a field name, and the Field is the associated field.
            filter_pred (callable or None): use only exanples for which
                filter_pred(example) is true, or use all examples if None.
                Default is None
        """
        self.examples = examples.apply(SeriesExample.fromSeries, args=(fields,), axis=1).tolist()
        if filter_pred is not None:
            self.examples = filter(filter_pred, self.examples)
        self.fields = dict(fields)
        # Unpack field tuples
        for n, f in list(self.fields.items()):
            if isinstance(n, tuple):
                self.fields.update(zip(n, f))
                del self.fields[n]


class SeriesExample(Example): # Thanks to https://stackoverflow.com/questions/52602071/dataframe-as-datasource-in-torchtext
    """Class to convert a pandas Series to an Example"""
    @classmethod
    def fromSeries(cls, data, fields):
        return cls.fromdict(data.to_dict(), fields)
    @classmethod
    def fromdict(cls, data, fields):
        ex = cls()
        for key, field in fields.items():
            if key not in data:
                raise ValueError("Specified key {} was not found in "
                "the input data".format(key))
            if field is not None:
                setattr(ex, key, field.preprocess(data[key]))
            else:
                setattr(ex, key, data[key])
        return ex


class BERTGRU(torch.nn.Module): # Thanks to https://github.com/bentrevett/pytorch-sentiment-analysis/blob/master/6+-+Transformers+for+Sentiment+Analysis.ipynb
    def __init__(self,
                 bert,
                 hidden_dim,
                 output_dim,
                 n_layers,
                 bidirectional,
                 dropout):
        super().__init__()        
        self.bert = bert       
        embedding_dim = bert.config.to_dict()['hidden_size']        
        self.rnn = torch.nn.GRU(embedding_dim,
                          hidden_dim,
                          num_layers = n_layers,
                          bidirectional = bidirectional,
                          batch_first = True,
                          dropout = 0 if n_layers < 2 else dropout)        
        self.out = torch.nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)        
        self.dropout = torch.nn.Dropout(dropout)
        

    def forward(self, text):
        #text = [batch size, sent len]
        with torch.no_grad():
            embedded = self.bert(text)[0]
        #embedded = [batch size, sent len, emb dim]
        _, hidden = self.rnn(embedded)
        #hidden = [n layers * n directions, batch size, emb dim]
        if self.rnn.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
        #hidden = [batch size, hid dim]
        output = self.out(hidden)
        #output = [batch size, out dim]
        return output


def build(matrix, class_names):
    binary = np.array(matrix)
    fig, ax = plot_confusion_matrix(conf_mat=binary,
                show_absolute=True,
                show_normed=False,
                colorbar=True,
                class_names=class_names)
    plt.savefig('confusion_matrix_'+'_'.join(class_names)+'.png')


def read_data():
    """ Returns training, validation and test data"""
    data_df = pd.read_csv("imdb_movie_reviews.csv", names=["text", "label"]) #Column 1: any texts, column 2: any binary labels
#    data_df['text'] = data_df['text'].apply(lambda x: remove_breaks(x)) # Remove breaks
    train_df = data_df[1:20001]
    valid_df = data_df[20001:25001]
    test_df = data_df[25001:37501]
    return train_df, valid_df, test_df


def tokenize_and_cut(sentence):
    """ Returns tokenized and cut sentence with (max length -2) because of start and end token"""
    tokens = tokenizer.tokenize(sentence) 
    tokens = tokens[:max_input_length-2]
    return tokens


def count_parameters(model):
    """ Returns number of parameters in the model """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def accuracy(preds, y):
    """ Returns accuracy per batch """
    predlist = []
    preds = preds.tolist()
    for p in preds:
        predlist.append(p.index(max(p)))
    ylist = []
    y = y.tolist()
    for ij in y:
        ylist.append(ij.index(max(ij)))
    acc = accuracy_score(predlist,ylist)
    return acc


def train(model, iterator, optimizer, criterion):
    """ Returns training loss and accuracy after training, per epoch """
    epoch_loss = 0
    epoch_acc = 0
    model.train()
    for batch in iterator:        
        optimizer.zero_grad()        
        predictions = model(batch.text).squeeze(1)
        y_onehot = batch.label.cpu().numpy()
        y_onehot = (np.arange(nr_classes) == y_onehot[:,None]).astype(np.float32)
        y_onehot = torch.from_numpy(y_onehot)
        batch.label=y_onehot.to(device)
        loss = criterion(predictions, batch.label)
        acc = accuracy(predictions, batch.label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    """ Returns validation loss and accuracy after validating, per epoch """
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
    with torch.no_grad():
        for batch in iterator:
            predictions = model(batch.text).squeeze(1)
            y_onehot = batch.label.cpu().numpy()
            y_onehot = (np.arange(nr_classes) == y_onehot[:,None]).astype(np.float32)
            y_onehot = torch.from_numpy(y_onehot)
            batch.label=y_onehot.to(device)
            loss = criterion(predictions, batch.label)
            acc = accuracy(predictions, batch.label)
            epoch_loss += loss.item()
            epoch_acc += acc.item()
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def epoch_time(start_time, end_time):
    """ Returns past time for one epoch """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def predict_sentiment(model, tokenizer, review):
    """ Returns prediction of sentiment """
    model.eval()
    tokens = tokenizer.tokenize(review)
    tokens = tokens[:max_input_length-2]
    indexed = [init_token_idx] + tokenizer.convert_tokens_to_ids(tokens) + [eos_token_idx]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(0)
    prediction = torch.sigmoid(model(tensor))
    return prediction

def remove_breaks(string):
    """ Returns string with break tokens removed """
    try:
        string = string.replace("<br />", "")
    except:
        string = string
    return string


def sentiment(pretrained_model, train_df, valid_df, test_df, params, training):
    """ Performs sentiment analysis """
    # Set random seeds
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnndeterministic = True

    # Set tokenizer
    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(pretrained_model)
#    print(len(tokenizer.vocab)) # Prints number of tokens in vocabulary

    # Set special tokens
    init_token = tokenizer.cls_token
    eos_token = tokenizer.sep_token
    pad_token = tokenizer.pad_token
    unk_token = tokenizer.unk_token
    global init_token_idx
    init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
    global eos_token_idx
    eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
    pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
    unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

    # Set maximum input length
    global max_input_length
    max_input_length = tokenizer.max_model_input_sizes[pretrained_model]
#    print(max_input_length) # Prints maximum input length

    # Set text field
    text = Field(batch_first = True,
                      use_vocab = False,
                      tokenize = tokenize_and_cut,
                      preprocessing = tokenizer.convert_tokens_to_ids,
                      init_token = init_token_idx,
                      eos_token = eos_token_idx,
                      pad_token = pad_token_idx,
                      unk_token = unk_token_idx)
    # Set label field
    label = Field(dtype = torch.float, sequential=False, unk_token=None)

    # Set data
    fields = { 'label' : label, 'text' : text }
    train_data = DataFrameDataset(train_df, fields)
    valid_data = DataFrameDataset(valid_df, fields)
    if training:
        print(f"Number of training instances: {len(train_df)}") # Prints number of training instances
        print(f"Number of validation instances: {len(valid_df)}") # Prints number of validation instances
    print(f"Number of testing instances: {len(test_df)}") # Prints number of testing instances

#    print(vars(train_data.examples[6])) # Prints a training instance example to check that text is numericalized

#    print(tokenizer.convert_ids_to_tokens(vars(train_data.examples[6])['text'])) # Prints the text of the training instance example

    # Build vocabulary for the labels
    label.build_vocab(train_data)
#    print(label.vocab.stoi) # Prints the vocabulary for the labels
    names = [] # Specify names for negative and positive used in .csv file respecively
    for l in label.vocab.stoi:
        names.append(l)

    # Set name of model to be loaded or created
    if 'dutch' in pretrained_model:
        load_model = 'bert_gru_sentiment_dutch_'+"_".join(names)+'.pt'
    else:
        load_model = 'bert_gru_sentiment_english_'+"_".join(names)+'.pt'
    
    if training:
        print("Prepare for training and testing...")
    else:
        print("Prepare for testing with", load_model+"...")
        
    # Set parameters
    global nr_classes
    nr_classes = len(names)
    params['output_dim'] = nr_classes
    hidden_dim = params['hidden_dim']
    output_dim = params['output_dim']
    n_layers = params['n_layers']
    bidirectional = params['bidirectional']
    dropout = params['dropout']
    n_epochs = params['n_epochs']
    batch_size = params['batch_size']

    # Set device
    global device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set iterators
    train_iterator, valid_iterator = BucketIterator.splits(
        (train_data, valid_data), 
        batch_size = batch_size, 
        device = device,
        sort_key=lambda x: len(x.text),
        sort_within_batch=False)

    # Set pre-trained model
    print("Applying pre-trained", pretrained_model, "model, feeding into GRU...")
    bert = BertModel.from_pretrained(pretrained_model)
#    print(bert) # Prints the BERT model
    
    # Set model
    model = BERTGRU(bert,
                    hidden_dim,
                    output_dim,
                    n_layers,
                    bidirectional,
                    dropout)

    # Print parameters
#    print(f'The model has {count_parameters(model):,} parameters') # Prints the total number of parameters
#    for name, param in model.named_parameters():                
#        if name.startswith('bert'):
#            param.requires_grad = False
#    print(f'The model has {count_parameters(model):,} trainable parameters') # Prints the total number of parameters without the BERT Transformer, so only GRU
#    for name, param in model.named_parameters():                
#        if param.requires_grad:
#            print(name) #Prints the trainable parameters


    # Set optimizer, criterion, place model and criterion on GPU if available
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss()
    model = model.to(device)
    criterion = criterion.to(device)
    
    # Set initial best validation loss to infinity
    best_valid_loss = float('inf')

    # Train the model for n_epochs
    if training:
        print("Starting training...")
        for epoch in range(n_epochs):
            start_time = time.time()
            train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                # Save best model
                torch.save(model.state_dict(), load_model)

            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s') # Prints the current epoch number and time
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%') # Prints the current training loss and accuracy
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%') # Prints the current validation loss and accuracy

    # Load best model for testing
    print("Loading best model for testing...")
    model.load_state_dict(torch.load(load_model, map_location=torch.device('cpu')))

    # Transform test data for classification report
    transformer_dict = {}
    for nr, lab in enumerate(names):
        transformer_dict[lab] = nr
    test_df['label'] = test_df['label'].apply(lambda x: transformer_dict[x])
    reviews = test_df['text'].tolist()
    labels = test_df['label'].tolist()
    # Testing
    print("Testing...")
    predlist = []
    large_error_count = 0
    start_time = time.time()
    for nr,review in enumerate(reviews):
        pred = predict_sentiment(model, tokenizer, review)
        pred = pred.tolist()
        pred = pred[0]
        p = pred.index(max(pred))
        predlist.append(p)
        if abs(int(p) - int(labels[nr])) > 1:
            large_error_count += 1
        if p != int(labels[nr]):
            print(names[int(labels[nr])], "\t", names[p], "\t", review, "\n") # Prints mistakes
    end_time = time.time()
    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    # Show test results
    print(f'Testing Time: {epoch_mins}m {epoch_secs}s') # Prints testing time
    print(classification_report(predlist, labels, target_names=names)) # Prints classification report
    cm = confusion_matrix(predlist, labels)
    print(cm) # Prints confusion matrix
    build(cm, names) # Save confusion matrix as image
    print(params) # Prints parameters
#    print("Errors > 1:", large_error_count, "out of", len(test_df), "accuracy:", (len(test_df)-large_error_count)/len(test_df)) # Prints > 1 miss-class 


def main():
    print("Reading data...")
    train_df, valid_df, test_df = read_data()
    print("Done reading data")
    # Set name of pre-trained model
    language = detect(train_df['text'].tolist()[0])
    if language == 'en':
        pretrained_model = 'bert-base-cased'
    elif language == 'nl':
        pretrained_model = 'bert-base-dutch-cased'
    else:
        print("Language not detected, specify pre-trained model in code")
    print("Using", pretrained_model)
    # Set training = False for testing only
    training = True
    # Set parameters
    params = {  'hidden_dim' : 256,
                'n_layers' : 2,
                'bidirectional' : True,
                'dropout' : 0.25,
                'n_epochs' : 3,
                'batch_size' : 128
            }

    # Run
    sentiment(pretrained_model, train_df, valid_df, test_df, params, training)


if __name__ == '__main__':
    main()