import torch
import json
from torch.utils.data import Dataset

class ICEWSData(Dataset):
    def __init__(self,subset,fourteen,gpuNr):
        self.subset = subset
        if fourteen:
            self.baseDataPath = "../icews14/"
        else: 
            self.baseDataPath = "../icews05-15/"
        
        with open(f'{self.baseDataPath}{subset}.json') as f:
            split = json.load(f)
        self.dataSplit = split
        self.dates = sorted(list(split.keys()))
        self.gpuNr = gpuNr
        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')

    def __getitem__(self,x):
        triples = torch.LongTensor(self.dataSplit[self.dates[x]])
        entities = triples[torch.arange(0,triples.shape[0],3)]
        relations = triples[torch.arange(1,triples.shape[0],3)]
        targets = triples[torch.arange(2,triples.shape[0],3)]
        return entities.to(self.device), relations.to(self.device), targets.to(self.device)
    
    def __len__(self):
        return len(list(self.dataSplit.keys()))

if __name__ == "__main__":
    dataset = ICEWSData('test',True,0)
    print (dataset[5])