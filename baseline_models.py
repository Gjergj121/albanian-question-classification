import fasttext

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate

import re
from constants import *
from utils import *


if __name__ == '__main__':
    ft = fasttext.load_model('data/cc.sq.300.bin')
    dataframe = pd.read_csv('data/translated_dataset_manuale.csv')
    dataframe = dataframe[dataframe.columns[1:]] 

    pattern = re.compile(r'\b(' + r'|'.join(STOPWORDS) + r')\b\s*')
    indexToLabel = {}
    indexToEmbedding = {}

    for index, row in dataframe.iterrows():
        text = pattern.sub('', row['Translation'][:-1])
        indexToEmbedding[index] = ft.get_sentence_vector(text)
        indexToLabel[index] = row['Label Coarse']
        
        
    indeces = list(indexToLabel.keys())
    labels = list(indexToLabel.values())
    train_idx, test_idx, train_labels, test_labels = train_test_split(indeces, labels, test_size=0.2, 
                                                                    random_state=SEED)
    
    train_features = get_features(train_idx, indexToEmbedding)
    test_features = get_features(test_idx, indexToEmbedding)
    
    models = [LinearSVC(max_iter=1000), SVC(max_iter=1000),RandomForestClassifier(n_estimators=100),
          LogisticRegression(max_iter=1000)]

    for model in models:
        print(model)
        model.fit(np.array(train_features), np.array(train_labels))
        pred = model.predict(test_features)
        metrics = get_metrics(test_labels, pred)
        print(metrics)
        print('#' * 80)
        
    # To run cross validations
    
    # all_features = np.array(get_features(indeces, indexToEmbedding))
    # models = [LinearSVC(max_iter=1000), SVC(max_iter=1000),RandomForestClassifier(n_estimators=100),
    #         LogisticRegression(max_iter=1000)]

    # for model in models:
    #     cv_results = cross_validate(model, all_features, labels, cv=5)
    #     print(cv_results['test_score'])
        