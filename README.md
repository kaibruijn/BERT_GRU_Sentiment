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

# Baselines (SVM, Logistic Regression and Linear Regression)
Sentiment Analysis using traditional machine learning methods for Language Technology Project at University of Groningen

Run (train + test):
Simply run [baseline].py, where [baseline] is replaced with either 'svm', 'linear_regression' or 'logistic_regression'. 

Change data:
read_data_baselines.py provides possible data sets. For this, simply change read_data() in [baseline].py.

If you want to train on the binary sentiment analysis task and test on multiclass sentiment analysis, run trainbinary_testmulti_[baseline]_multiclass_movies, where [baseline] is replaced with a traditional method. Here, the read_data() cannot be changed.

A basic simpletransformers [1] implementation using 'bert-base-cased' with default settings is also added as a baseline. It can be run on the IMDB Movie data set using simpletransformer_imdb.py

# Requirements:
- langdetect==1.0.8
- matplotlib==3.2.1
- mlxtend==0.17.2
- nltk==3.4.5
- numpy==1.18.4
- pandas==0.25.3
- pycuda==2019.1.2
- simpletransformers==0.10.2
- torch==1.3.1
- torchtext==0.6.0
- transformers==2.9.1

# File desciption:
- bert_gru.py - main file for doing classification
- bert_gru.sh - shell script used on University of Groningen's Peregrine HPC Cluster
- dutch_movie_reviews_test.csv - data set for Dutch binary movie reviews [2]
- dutch_movie_reviews_train.csv - data set for Dutch binary movie reviews [2]
- emotion_classification.txt - data set for English 13-class emotion classification [3]
- five_star_movie_reviews.tsv - data set for English five star movie reviews [4]
- imdb_movie_reviews.csv - data set for English binary movie reviews [5][6]
- linear_regression.py- linear regression baseline for classification
- logistic_regression.py- logistic regression baseline for classification
- read_data_baselines.py - read_data() functions fot the baselines for all data sets
- read_datas.py - read_data() functions for all data sets
- six_class_reviews.txt - data set for English 6-class clasification of product reviews [7][8]
- svm.py - svm baseline for classification
- trainbinary_testmulti_svm_multiclass_movies.py - svm for training on binary task and testing on multiclass task
- trainbinary_testmulti_lin_reg_multiclass_movies.py - linear regression for training on binary task and testing on multiclass task
- trainbinary_testmulti_logistic_regression_multiclass_movies.py - logistic regression for training on binary task and testing on multiclass task

# References:
1. https://github.com/ThilinaRajapakse/simpletransformers/
2. https://github.com/benjaminvdb/110kDBRD
3. https://data.world/crowdflower/sentiment-analysis-in-text
4. https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
5. https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/data
6. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011, June). Learning word vectors for sentiment analysis. In Proceedings of the 49th annual meeting of the association for computational linguistics: Human language technologies-volume 1 (pp. 142-150). Association for Computational Linguistics.
7. https://www.kaggle.com/jeromeblanchet/multidomain-sentiment-analysis-dataset/data
8. Blitzer, J., Dredze, M., & Pereira, F. (2007, June). Biographies, bollywood, boom-boxes and blenders: Domain adaptation for sentiment classification. In Proceedings of the 45th annual meeting of the association of computational linguistics (pp. 440-447).
