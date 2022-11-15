from loss_fct import loss_fn
from utils import *
import torch.optim as optim
import os
from tqdm import tqdm
from models import Attention, BiLSTM, TransformerModel
import fasttext
import collections
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
import re
from constants import *
import pandas as pd
import torch.nn as nn
import logging 
import numpy as np
import json
from argparse import ArgumentParser


TIMESTAMP = get_current_timestamp()

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
    
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"logs/{TIMESTAMP}.log"),
        logging.StreamHandler()
    ]
)



def yield_tokens(data_iter):
    for _, text, _ in data_iter:
        yield tokenizer(text)


def collate_batch(batch):
    label_list, text_list, index_list = [], [], []
    for(_label, _text, _index) in batch:
        label_list.append(label_pipeline(_label))
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        index_list.append(int(_index))
        
    label_list = torch.tensor(label_list, dtype=torch.int64)
    text_list = pad_sequence(text_list, padding_value=0)
    return text_list, label_list, index_list


# python train_models.py --num_epochs=200 --learning_rate=1e-4 --loss_type='focal' --model_name='transformer' --batch_size=64 --use_attention='true'

parser = ArgumentParser()
parser.add_argument("--model_name", dest="model_name", required=True, type=str) # ['transformer', 'bilstm']
parser.add_argument("--use_attention", dest="use_attention", default='true', type=str2bool)
parser.add_argument("--num_epochs", dest="num_epochs", default=100, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-3, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='focal', type=str)
parser.add_argument("--results_dir", dest="results_dir", type=str, default='results')


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    model_name = args.model_name
    use_attention = args.use_attention
    loss_type = args.loss_type
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    results_dir = args.results_dir
    batch_size = args.batch_size
    assert model_name in {'transformer', 'bilstm'}, 'Wrong model name'
    
    all_examples, all_labels = get_all_examples()
    num_labels = len(set(all_labels))

    tokenizer = lambda s: s.split()

    vocab = build_vocab_from_iterator(yield_tokens(all_examples), specials=["<pad>", "<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)
    
    
    train_data, test_data, train_labels, test_labels = train_test_split(all_examples, all_labels, test_size=0.2, 
                                                                                random_state=SEED)

    train_data, val_data, train_labels, val_labels = train_test_split(train_data, train_labels, test_size=0.15, 
                                                                                random_state=SEED)
    
    train_size_stats = "Training Size: {}, Labels Stats {}".format(len(train_data), collections.OrderedDict(sorted(collections.Counter(train_labels).items())))
    logging.info(train_size_stats)
    val_size_stats = "Validation Size: {}, Labels Stats {}".format(len(val_data), collections.OrderedDict(sorted(collections.Counter(val_labels).items())))
    logging.info(val_size_stats)
    test_size_stats = "Test Size: {}, Labels Stats {}".format(len(test_data), collections.OrderedDict(sorted(collections.Counter(test_labels).items())))
    logging.info(test_size_stats)
    
    train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, collate_fn = collate_batch)
    val_dataloader = DataLoader(val_data, batch_size = batch_size, shuffle = False, collate_fn = collate_batch)
    test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False, collate_fn = collate_batch)
    
    ft = fasttext.load_model('data/cc.sq.300.bin')
    
    embedding_layer = get_embedding_layer(vocab, ft)

    if model_name == 'transformer':
        model = TransformerModel(embedding_layer, 300, 4, 300, num_labels, 2).to(DEVICE)
    elif model_name == 'bilstm':
        model = BiLSTM(embedding_layer, 300, num_labels, use_attention=use_attention, bidirectional=True).to(DEVICE)
    else:
        raise Exception('Wrong model name')
    
    # Training part
    clip = 5
    samples_per_class_train = get_samples_per_class(torch.tensor(train_labels))
    num_training_steps = num_epochs * len(train_dataloader)
    progress_bar = tqdm(range(num_training_steps))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    checkpoint_dir = os.path.join(f'results/best_models/{model_name}_{TIMESTAMP}_best_model_sampled.pt')
    
    best_f1 = 0
    best_val = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            input_text = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)
            indeces = batch[2]
            output = model(input_text)
        
            loss = loss_fn(output, labels, no_of_classes=num_labels, samples_per_cls=samples_per_class_train, loss_type=loss_type)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_metrics = evaluate(val_dataloader, model)
        val_metrics.pop('cm')
        if (epoch +1) % 10 == 0:
            logging.info("Epoch {} **** Loss {} **** Metrics validation: {}".format(epoch + 1, loss, val_metrics))
        
        if val_metrics['macro'] > best_f1:
            best_f1 = val_metrics['macro']
            best_val = val_metrics
            torch.save(model.state_dict(), checkpoint_dir)
            

    logging.info("Evaluating")
    model.load_state_dict(torch.load(checkpoint_dir))
    model.to(DEVICE)
        
    test_metrics = evaluate(test_dataloader, model, True)
    results = test_metrics.pop('results')
    logging.info(test_metrics)
    
    result_logs = {'id': TIMESTAMP}
    result_logs['seed'] = SEED
    result_logs['model_name'] = model_name
    result_logs['use_attention'] = use_attention
    result_logs['train_stats'] = train_size_stats
    result_logs['val_stats'] = val_size_stats
    result_logs['test_stats'] = test_size_stats
    result_logs['epochs'] = num_epochs
    result_logs['batch_size'] = batch_size
    result_logs['optimizer'] = optimizer.defaults
    result_logs["loss_type"] = loss_type
    result_logs['best_validation_metrics'] = best_val
    result_logs['test_metrics'] = test_metrics
    result_logs['checkpoint_dir'] = checkpoint_dir
    result_logs['results'] = results
    
    res_file = os.path.join(results_dir, f'{model_name}_{TIMESTAMP}' + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)