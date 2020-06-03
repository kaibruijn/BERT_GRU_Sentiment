# BERT_GRU_Sentiment
Sentiment Analysis using BERT and GRU for Language Technology Project at University of Groningen

The model will automatically recognize the language. A GRU will be stacked on BERT representations and perform classification.

Run (train + test):
Simply run bert_gru.py.

Run (test):
If a trained model (.pt file) exists, set training = False in the code. Then run bert_gru.py.

Change data:
read_datas.py provides possible data sets. For this, simply change read_data() in bert_gru.py.
New data sets need to be in a Pandas DataFrame as can be seen in read_datas.py.

Requirements:
- langdetect==1.0.8
- matplotlib==3.2.1
- mlxtend==0.17.2
- numpy==1.18.4
- pandas==0.25.3
- pycuda==2019.1.2
- torch==1.3.1
- torchtext==0.6.0
- transformers==2.9.1

