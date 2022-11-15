import collections
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from models import BertClassifier
from utils import *
from transformers import AutoTokenizer, BertTokenizerFast, DataCollatorWithPadding, AdamW, get_scheduler
from datasets import DatasetDict, Dataset, Features, Value
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import logging
from tqdm import tqdm
from utils import *
from loss_fct import * 
import os


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


def evaluate_bert(dataloader, model, return_predictions=False):
    model.eval()
    all_pred = ['predictions']
    all_labels = ['gold']
    all_ids = ['text_indeces']
    
    for batch in dataloader:
        indeces = batch.pop("index")
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        labels = batch.pop("labels")
    
        with torch.no_grad():
            output = model(batch)

        predictions = torch.argmax(output, dim=-1)
    
        all_pred.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_ids.extend(indeces.cpu().numpy())
    
    if return_predictions:
        metrics = get_metrics(all_labels[1:], all_pred[1:])
        metrics['results'] = list(zip(all_ids, all_pred, all_labels))
        return metrics

    return get_metrics(all_labels[1:], all_pred[1:])


parser = ArgumentParser()
parser.add_argument("--num_epochs", dest="num_epochs", default=100, type=int)
parser.add_argument("--learning_rate", dest="learning_rate", default=1e-4, type=float)
parser.add_argument("--batch_size", dest="batch_size", default=64, type=int)
parser.add_argument("--loss_type", dest="loss_type", default='focal', type=str)
parser.add_argument("--results_dir", dest="results_dir", type=str, default='results')


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args, logging)
    loss_type = args.loss_type
    num_epochs = args.num_epochs
    learning_rate = args.learning_rate
    results_dir = args.results_dir
    batch_size = args.batch_size
    checkpoint_dir = os.path.join(f'results/best_models/bert_{TIMESTAMP}_best_model_sampled.pt')

    all_examples, all_labels = get_all_examples()
    num_labels = len(set(all_labels))
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
    
    raw_dataset = get_raw_dataset(train_data, val_data, test_data)

    ds = DatasetDict()

    for split, d in raw_dataset.items():
        ds[split] = Dataset.from_dict(mapping=d, features=Features({'label': Value(dtype='int64'), 
                                                                        'text': Value(dtype='string'), 'index': Value(dtype='int64')}))

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    def tokenize_function(example):
        return tokenizer(example["text"], truncation=True, max_length=256)

    tokenized_dataset = ds.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    tokenized_dataset = tokenized_dataset.remove_columns(["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    tokenized_dataset.set_format("torch")

    train_dataloader = DataLoader(
        tokenized_dataset["train"], batch_size=batch_size, collate_fn=data_collator, shuffle = True
    )
    eval_dataloader = DataLoader(
        tokenized_dataset["val"], batch_size=batch_size, collate_fn=data_collator
    )

    test_dataloader = DataLoader(
        tokenized_dataset["test"], batch_size=batch_size, collate_fn=data_collator
    )
    
    model = BertClassifier(num_labels, roberta='roberta-base').to(DEVICE)

    optimizer = AdamW(model.parameters(), lr=learning_rate)

    num_training_steps = num_epochs * len(train_dataloader)
    samples_per_class_train = get_samples_per_class(tokenized_dataset["train"]['labels'])

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    progress_bar = tqdm(range(num_training_steps))
    best_accuracy = 0
    best_f1 = 0
    val_metrics = []
    train_loss = []
    
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            indeces = batch.pop("index")
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            output = model(batch)
        
            loss = loss_fn(output, labels, no_of_classes=num_labels, samples_per_cls=samples_per_class_train, loss_type=loss_type)
            
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        val_metrics = evaluate_bert(eval_dataloader, model)
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
        
    test_metrics = evaluate_bert(test_dataloader, model, True)
    results = test_metrics.pop('results')
    logging.info(test_metrics)
    
    result_logs = {'id': TIMESTAMP}
    result_logs['seed'] = SEED
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
    
    res_file = os.path.join(results_dir, f'bert_{TIMESTAMP}' + ".json")
    with open(res_file, mode='w') as f:
        json.dump(result_logs, f, cls=NpEncoder, indent=2)