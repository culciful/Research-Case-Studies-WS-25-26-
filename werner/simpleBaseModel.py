import torch
import torch.nn as nn
import simpleDataLoader as dl
import torch.optim as optim

class LinkPredictor(nn.Module):
    def __init__(self,nrRelations,nrEntities,dimEmbedding,gpuNr):
        super(LinkPredictor,self).__init__()
        self.nrRelations = nrRelations
        self.nrEntities = nrEntities
        self.dimEmbedding = dimEmbedding
        self.gpuNr = gpuNr
        self.device = torch.device('cuda:'+str(self.gpuNr) if torch.cuda.is_available() else 'cpu')

        self.entities = nn.Embedding(self.nrEntities,self.dimEmbedding).to(self.device)
        nn.init.xavier_uniform_(self.entities.weight)
        self.relations = nn.Embedding(self.nrRelations,self.dimEmbedding).to(self.device)
        nn.init.xavier_uniform_(self.relations.weight)
        self.scorer = nn.Linear(2*self.dimEmbedding,self.nrEntities).to(self.device) #learnable scoring function
        nn.init.xavier_uniform_(self.scorer.weight)

    def forward(self,input):
        entities = self.entities(input[0])
        relations = self.relations(input[1])
        scores = self.scorer(torch.cat([entities,relations],dim=1))
        return scores
    
if __name__ == '__main__':
    model = LinkPredictor(230,7128,300,0)
    ins = dl.ICEWSData('train',True,0)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(),lr=0.003,weight_decay=0.07)
    model.train()
    for x in range(5):
        print (f"Epoch {x+1}")
        absLoss = 0
        for sample in ins:
            scores = model(sample)
            targets = sample[-1]
            loss = lossFunction(scores,targets)
            loss.backward()
            optimizer.step()
            absLoss += float(loss)
            optimizer.zero_grad()
        print (absLoss/len(ins))
        #validation
        with torch.no_grad():
            insValid = dl.ICEWSData('valid',True,0)
            absLoss = 0
            for sample in insValid:
                scores = model(sample)
                targets = sample[-1]
                loss = lossFunction(scores,targets)
                absLoss += float(loss)

            print (f"Valid Loss:\t{absLoss/len(insValid)}")
    #testing
    with torch.no_grad():
        insTest = dl.ICEWSData('test',True,0)
        hits1 = 0
        hits3 = 0
        hits10 = 0
        ranks = 0
        rRanks = 0
        sampleCount = 0
        for sample in insTest:
            scores = model(sample)
            targets = sample[-1]
            for i in range(scores.shape[0]):
                sorted_scores = torch.sort(scores[i], descending=True).values
                target_score = scores[i][targets[i]]
                rank = (sorted_scores == target_score).nonzero()[0, 0].item() + 1

                ranks += rank
                rRanks += 1/rank
                sampleCount += 1
                if targets[i] in torch.topk(scores[i],1).indices:
                    hits1 += 1
                if targets[i] in torch.topk(scores[i],3).indices:
                    hits3 += 1
                if targets[i] in torch.topk(scores[i],10).indices:
                    hits10 += 1
        print ("~~~~Test Results~~~~\n")
        print (f"Hits@1:\t\t{hits1/sampleCount}")
        print (f"Hits@3:\t\t{hits3/sampleCount}")
        print (f"Hits@10:\t{hits10/sampleCount}")
        print (f"MR:\t\t{ranks/sampleCount}")
        print (f"MRR:\t\t{rRanks/sampleCount}")