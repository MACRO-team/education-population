import numpy as np
import random
import json

RANDOM_SEED = 7
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

def training_data(path) :
    dataset_mapping = json.load(open(path))

    train_set = []
    test_set = []

    for idx in dataset_mapping :
        if 'extract' in dataset_mapping[idx] :
            continue

        useful_lines = dataset_mapping[idx]['abstract'] + '. ' + dataset_mapping[idx]['participants']

        ## ignoring articles with no abstract and participants
        if useful_lines == '' :
            continue

        temp = {}
        temp['text'] = useful_lines
        temp['coding'] = dataset_mapping[idx]['edupopulation']
        temp["reference"] = dataset_mapping[idx]['reference']
        rand_int = random.randint(0,2)

        ## Spliting dataset into 80:20 train-test split
        if rand_int != 0 :
            train_set.append(temp)
        else :
            test_set.append(temp)

    return train_set, test_set



def training_data_sentences(path) :
    dataset_mapping = json.load(open(path))

    train_set = []
    test_set = []

    for idx in dataset_mapping :
        if 'extract' in dataset_mapping[idx] :
            continue

        if 'sentence_filter' not in dataset_mapping[idx] :
            continue
        sentence_lines = dataset_mapping[idx]['sentence_filter']

        useful_lines = ''

        for line in sentence_lines :
            useful_lines += line.strip() + '.'


        ## ignoring articles with no abstract and participants
        if useful_lines == '' :
            continue

        temp = {}
        temp['text'] = useful_lines
        temp['coding'] = dataset_mapping[idx]['edupopulation']

        rand_int = random.randint(0,2)

        ## Spliting dataset into 80:20 train-test split
        if rand_int != 0 :
            train_set.append(temp)
        else :
            test_set.append(temp)

    return train_set, test_set

