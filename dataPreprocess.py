import csv

import torch
import json

import pandas as pd

icews14PathTest = "icews14/icews_2014_test.txt"
icews14PathValid = "icews14/icews_2014_valid.txt"
icews14PathTrain = "icews14/icews_2014_train.txt"

icews0515PathTest = "icews05-15/icews_2005-2015_test.txt"
icews0515PathValid = "icews05-15/icews_2005-2015_valid.txt"
icews0515PathTrain = "icews05-15/icews_2005-2015_train.txt"


def readSets(fourteen):
    if fourteen:  # icews14
        test = pd.read_csv(icews14PathTest, sep="\t", header=None)
        valid = pd.read_csv(icews14PathValid, sep="\t", header=None)
        train = pd.read_csv(icews14PathTrain, sep="\t", header=None)
    else:  # icews05-15
        test = pd.read_csv(icews0515PathTest, sep="\t", header=None)
        valid = pd.read_csv(icews0515PathValid, sep="\t", header=None)
        train = pd.read_csv(icews0515PathTrain, sep="\t", header=None)
    return test, valid, train


def getEntitiesAndRelations(fourteen):  # heads + tails
    test, valid, train = readSets(fourteen)
    icews = pd.concat([test, train, valid])
    entitiesHead = icews[0].to_list()
    entitiesTail = icews[2].to_list()
    relationsTotal = set(icews[1].to_list())
    relationsTrain = set(train[1].to_list())
    allEnts = entitiesHead + entitiesTail
    entsTotal = set(allEnts)
    entsTrain = set(train[0].to_list() + train[2].to_list())
    timesteps = set(icews[3].to_list())
    return entsTotal, entsTrain, relationsTotal, relationsTrain, timesteps

def sortByTime(test, valid, train):
    test = test.sort_values(by=[3])
    valid = valid.sort_values(by=[3])
    train = train.sort_values(by=[3])
    return test, valid, train


def convert2id(set, output_file, fourteen):
    item2id = {}
    path_prefix = "icews14/" if fourteen else "icews05-15/"
    with open(f'{path_prefix + output_file}.txt', "w", encoding='utf-8') as file:
        for index, item in enumerate(set):
            item2id[item] = index
            text = item + '\t' + str(index) + '\n'
            file.write(text)
    return item2id


def data2id(fourteen):
    test, valid, train = readSets(fourteen)
    entsRels = getEntitiesAndRelations(fourteen)
    ent2id = convert2id(entsRels[0], 'entity2id', fourteen)
    rel2id = convert2id(entsRels[2], 'relation2id', fourteen)
    time2id = convert2id(entsRels[4], 'time2id', fourteen)

    for split in test,valid,train:
        split[0] = split[0].apply(lambda x: ent2id.get(x))
        split[2] = split[2].apply(lambda x: ent2id.get(x))
        split[1] = split[1].apply(lambda x: rel2id.get(x))
        split[3] = split[3].apply(lambda x: time2id.get(x))
    return test,valid,train


if __name__ == '__main__':
    '''fourteen = True
    path_prefix = "icews14/" if fourteen else "icews05-15/"
    entsRels = getEntitiesAndRelations(fourteen)

    test, valid, train = data2id(fourteen)

    for data, name in zip([test, train, valid], ['test', 'train', 'valid']):
        data.to_csv(f'{path_prefix + name}.csv', index=False)'''

    a = getEntitiesAndRelations(True)
    print(len(a[0]))
    print(len(a[2]))
    print(len(a[4]))

