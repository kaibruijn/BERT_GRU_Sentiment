from simpletransformers.classification import ClassificationModel
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

def main():
    data = read_data()
    train_df = transform_data(data[1:25000])
    eval_df = transform_data(data[37500:])
    args = transformer_arguments()
    architecture = 'bert'
    model_type = 'bert-base-cased'
    transformer(train_df, eval_df, architecture, model_type, args)

def transformer(train_df, eval_df, architecture, model_type, args):
    model = ClassificationModel(architecture, model_type, use_cuda=True, args=args)
    model.train_model(train_df)
    result, model_outputs, wrong_predictions = model.eval_model(eval_df, cr=classification_report, cm=confusion_matrix)
    for values in model_outputs:
    	print("P:\t", values[0], "\tN:\t", values[1])
    print(result['cr']) # Classification report
    print(result['cm']) # Confusion matrix

def read_data():
	return pd.read_csv("../Data/imdb_movie_reviews.csv", names=["text", "labels"])

def data_distribution(data):
	return Counter(data["labels"].tolist())

def transform_data(data):
    data = data[["labels", "text"]]
    transformer_dict = {"negative": 0, "positive": 1}
    data['labels'] = data['labels'].apply(lambda x: transformer_dict[x])
    return data
    
def transformer_arguments():
    return {
      'output_dir': 'output/',
      'cache_dir': 'cache/',
      'max_seq_length': 128,
      'train_batch_size': 8,
      'eval_batch_size': 8,
      'gradient_accumulation_steps': 1,
      'num_train_epochs': 1,
      'weight_decay': 0,
      'learning_rate': 4e-5,
      'adam_epsilon': 1e-8,
      'warmup_ratio': 0.06,
      'warmup_steps': 0,
      'max_grad_norm': 1.0,
      'fp16': False,

      'logging_steps': 50,
      'evaluate_during_training': False,
      'save_steps': 2000,
      'eval_all_checkpoints': True,
      'use_tensorboard': True,

      'overwrite_output_dir': True,
      'reprocess_input_data': False,
    }

if __name__ == '__main__':
    main()