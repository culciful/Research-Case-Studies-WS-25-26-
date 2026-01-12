import torch
import json

import pandas as pd

icews14PathTest = "../icews14/icews_2014_test.txt"
icews14PathValid = "../icews14/icews_2014_valid.txt"
icews14PathTrain = "../icews14/icews_2014_train.txt"

icews0515PathTest = "../icews05-15/icews_2005-2015_test.txt"
icews0515PathValid = "../icews05-15/icews_2005-2015_valid.txt"
icews0515PathTrain = "../icews05-15/icews_2005-2015_train.txt"

def readSets(fourteen):
    if fourteen: #icews14
        test = pd.read_csv(icews14PathTest,sep="\t",header=None)
        valid = pd.read_csv(icews14PathValid,sep="\t",header=None)
        train = pd.read_csv(icews14PathTrain,sep="\t",header=None)
    else: #icews05-15
        test = pd.read_csv(icews0515PathTest,sep="\t",header=None)
        valid = pd.read_csv(icews0515PathValid,sep="\t",header=None)
        train = pd.read_csv(icews0515PathTrain,sep="\t",header=None)
    return test,valid,train

def getEntitiesAndRelations(fourteen): #heads + tails
    test,valid,train = readSets(fourteen)
    icews = pd.concat([test,train,valid])
    entitiesHead = icews[0].to_list()
    entitiesTail = icews[2].to_list()
    relationsTotal = set(icews[1].to_list())
    relationsTrain = set(train[1].to_list())
    allEnts = entitiesHead + entitiesTail
    entsTotal = set(allEnts)
    entsTrain = set(train[0].to_list()+train[2].to_list())
    timesteps = set(icews[3].to_list())
    return entsTotal, entsTrain, relationsTotal, relationsTrain, timesteps

def entRel2id(entityset,relationset):
    ent2id = {}
    rel2id = {}
    for i, entity in enumerate(entityset):
        ent2id[entity] = i
    for j, relation in enumerate(relationset):
        rel2id[relation] = j
    return ent2id,rel2id

def sortByTime(test,valid,train):
    test = test.sort_values(by=[3])
    valid = valid.sort_values(by=[3])
    train = train.sort_values(by=[3])
    return test,valid,train

def data2id(fourteen,extrapolation): #which set True for 14, False for 0515 | False for interpolation, True for extrapolation
    test,valid,train = readSets(fourteen)
    if extrapolation:
        test,valid,train = sortByTime(test,valid,train)
    entsRels = getEntitiesAndRelations(fourteen)
    ent2id, rel2id = entRel2id(entsRels[0],entsRels[2])

    for split in test,valid,train:
        split[0] = split[0].apply(lambda x: ent2id.get(x))
        split[2] = split[2].apply(lambda x: ent2id.get(x))
        split[1] = split[1].apply(lambda x: rel2id.get(x))
    return test,valid,train

def id2tensorDict(idSubset):
    heads = idSubset[0].to_list()
    rels = idSubset[1].to_list()
    tails = idSubset[2].to_list()
    times = idSubset[3].to_list()
    tensorDict = {}
    for h,r,t,ti in zip(heads,rels,tails,times):
        if not ti in tensorDict.keys():
            tensorDict[ti] = []
        else:
            tensorDict[ti].append(h)
            tensorDict[ti].append(r)
            tensorDict[ti].append(t)
    return tensorDict


if __name__ == '__main__':
    '''test, valid, train = data2id(False,True)
    #test,valid,train = readSets(True)
    test = id2tensorDict(test)
    train = id2tensorDict(train)
    valid = id2tensorDict(valid)
    for data,name in zip([test,train,valid],['test','train','valid']):
        with open(f'{name}.json', 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)'''
    a = getEntitiesAndRelations(True)
    print (len(a[0]))
    print (len(a[2]))
    # b = getEntitiesAndRelations(False)
    '''print (test)
    print (valid)
    print (train)'''
    #print (len(getEntitiesAndRelations(True)[-1]))