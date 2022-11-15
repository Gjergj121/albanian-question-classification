from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import torch 
import torch.nn as nn
import datetime
from constants import *
import json
import numpy as np
import argparse
import re
import pandas as pd



class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_raw_dataset(train_data, val_data, test_data):
    raw_dataset = {'train': {'index': [], 'text': [], 'label': []}, 
            'val': {'index': [], 'text': [], 'label': []}, 
            'test': {'index': [], 'text': [], 'label': []}}


    for label, text, index in train_data:
        raw_dataset['train']['index'].append(index)
        raw_dataset['train']['text'].append(text)
        raw_dataset['train']['label'].append(label)
        
    for label, text, index in val_data:
        raw_dataset['val']['index'].append(index)
        raw_dataset['val']['text'].append(text)
        raw_dataset['val']['label'].append(label)
        
    for label, text, index in test_data:
        raw_dataset['test']['index'].append(index)
        raw_dataset['test']['text'].append(text)
        raw_dataset['test']['label'].append(label)
        
    return raw_dataset


def get_all_examples():
    dataframe = pd.read_csv('data/translated_dataset_manuale.csv')
    dataframe = dataframe[dataframe.columns[1:]] 
    
    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    all_examples = []
    all_labels = []

    for index, row in dataframe.iterrows():
#        all_examples.append((row['Label Coarse'], pattern.sub('', row['Translation'][:-1]), index))
        all_examples.append((row['Label Coarse'], row['Text'][:-1], index))
        all_labels.append(int(row['Label Coarse']))
        
    return all_examples, all_labels
        

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    

def get_embedding_layer(vocab, ft):
    embedding_layer = nn.Embedding(len(vocab.get_stoi()), 300, padding_idx=vocab.get_stoi()['<pad>'])
    for word, index in vocab.get_stoi().items():
        with torch.no_grad():
            embedding_layer.weight[index] = torch.tensor(ft.get_word_vector(word))
    
    return embedding_layer


def get_metrics(gold, predictions):
    return {'macro': f1_score(gold, predictions, average='macro'), 'micro': f1_score(gold, predictions, average='micro'), 
            'weighted': f1_score(gold, predictions, average='weighted'), 
            'accuracy': accuracy_score(gold, predictions), 'cm': confusion_matrix(gold, predictions)}


def print_args(args, logger):
    for arg in vars(args):
        logger.info("{} \t \t {}".format(arg, getattr(args, arg)))
     

def get_features(indeces, embedder):
    features = list()
    for v in indeces:
        features.append(embedder[v])
        
    return features

        
def get_samples_per_class(labels):
    return torch.bincount(labels).tolist()


def get_current_timestamp():
    return str(datetime.datetime.now()).replace(" ", "_").replace(".", ":")


def evaluate(dataloader, model, return_predictions=False):
    model.eval()
    all_pred = ['predictions']
    all_labels = ['gold']
    all_ids = ['text_indeces']
    
    for batch in dataloader:
        input_text = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE)
        indeces = batch[2]
        
        with torch.no_grad():
            output = model(input_text)
        predictions = torch.argmax(output, dim=-1)
    
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(indeces)
    
    if return_predictions:
        metrics = get_metrics(all_labels[1:], all_pred[1:])
        metrics['results'] = list(zip(all_ids, all_pred, all_labels))
        return metrics

    return get_metrics(all_labels[1:], all_pred[1:])
