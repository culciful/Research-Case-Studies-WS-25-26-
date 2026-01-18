import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import QuadrupleDataLoader as dl
import torch.optim as optim


class TTransE(nn.Module):
    def __init__(self, nrRelations, nrEntities, nrTimes, dimEmbedding, margin=1.0):
        super(TTransE, self).__init__()
        self.nrRelations = nrRelations
        self.nrEntities = nrEntities
        self.nrTimes = nrTimes
        self.dimEmbedding = dimEmbedding
        self.margin = margin
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.entities = nn.Embedding(self.nrEntities, self.dimEmbedding)
        nn.init.xavier_uniform_(self.entities.weight)
        self.relations = nn.Embedding(self.nrRelations, self.dimEmbedding)
        nn.init.xavier_uniform_(self.relations.weight)
        self.times = nn.Embedding(self.nrTimes, self.dimEmbedding)
        nn.init.xavier_uniform_(self.times.weight)

    def generate_negative_samples(self, input):
        """Generate negative samples by corrupting subject and object separately"""
        subject_ids = input[0]
        object_ids = input[2]
        batch_size = subject_ids.size(0)

        subject_offset = torch.randint(1, self.nrEntities, (batch_size,), device=self.device)
        object_offset = torch.randint(1, self.nrEntities, (batch_size,), device=self.device)

        negative_subject_ids = (subject_ids + subject_offset) % self.nrEntities
        negative_object_ids = (object_ids + object_offset) % self.nrEntities
        return negative_subject_ids, negative_object_ids

    def forward(self, input):
        """Forward pass: compute distance scores"""
        subjects = self.entities(input[0])
        objects = self.entities(input[2])
        relations = self.relations(input[1])
        times = self.times(input[3])

        # TTransE: -||s + p + t - o||_2
        scores = -torch.norm(subjects + relations + times - objects, p=2, dim=1)  # p: L2
        return scores

    def compute_loss(self, input, positive_scores):
        """Compute Margin Ranking Loss"""
        negative_subject_ids, negative_object_ids = self.generate_negative_samples(input)

        # Corrupt head
        neg_input_h = [negative_subject_ids, input[1], input[2], input[3]]
        negative_scores_h = self(neg_input_h)

        # Corrupt tail
        neg_input_t = [input[0], input[1], negative_object_ids, input[3]]
        negative_scores_t = self(neg_input_t)

        # Combine negative samples
        negative_scores = torch.cat([negative_scores_h, negative_scores_t])
        positive_scores_expanded = positive_scores.repeat(2)

        # MarginRankingLoss: expect positive distance > negative distance
        loss_fn = nn.MarginRankingLoss(margin=self.margin, reduction='mean')
        labels = torch.ones_like(positive_scores_expanded)
        loss = loss_fn(positive_scores_expanded, negative_scores, labels)

        return loss

    def predict(self, subject_ids, relation_ids, time_ids):
        """
        Prediction: given (s, r, t), score all possible objects
        Returns: scores for all entities (lower distance is better)
        """
        batch_size = subject_ids.size(0)
        device = subject_ids.device

        # Get embeddings
        subjects = self.entities(subject_ids)  # (batch, dim)
        relations = self.relations(relation_ids)  # (batch, dim)
        times = self.times(time_ids)  # (batch, dim)

        # All possible objects
        all_objects = self.entities.weight  # (nrEntities, dim)

        # Compute scores for all possible objects
        # subjects: (batch, dim) -> (batch, 1, dim)
        # all_objects: (nrEntities, dim) -> (1, nrEntities, dim)
        subjects = subjects.unsqueeze(1)  # (batch, 1, dim)
        relations = relations.unsqueeze(1)  # (batch, 1, dim)
        times = times.unsqueeze(1)  # (batch, 1, dim)
        all_objects = all_objects.unsqueeze(0)  # (1, nrEntities, dim)

        # Broadcasting computation
        scores = torch.norm(subjects + relations + times - all_objects, p=2, dim=2)  # (batch, nrEntities)

        return scores

if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = TTransE(230, 7128, 365, 200, 1.0).to(device)
    train_dataset = dl.ICEWSData('train', True)
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    lossFunction = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.07)
    model.train()
    for x in range(40):
        print(f"Epoch {x + 1}")
        absLoss = 0
        num_batches = 0
        for sample in train_loader:
            sample = [s.to(device) for s in sample]

            optimizer.zero_grad()
            scores = model(sample)
            loss = model.compute_loss(sample, scores)

            loss.backward()
            optimizer.step()

            absLoss += loss.item()
            num_batches += 1
        print(absLoss / num_batches)
        # validation
        model.eval()
        with torch.no_grad():
            valid_dataset = dl.ICEWSData('valid', True)
            valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True)
            absLoss = 0
            num_batches = 0
            for sample in valid_loader:
                sample = [s.to(device) for s in sample]

                scores = model(sample)
                loss = model.compute_loss(sample, scores)

                absLoss += loss.item()
                num_batches += 1

            print(f"Valid Loss:\t{absLoss / num_batches}")
    # testing
    model.eval()
    with torch.no_grad():
        test_dataset = dl.ICEWSData('test', True)
        test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

        hits1 = 0
        hits3 = 0
        hits10 = 0
        total_rank = 0
        total_reciprocal_rank = 0
        sample_count = 0

        for sample in test_loader:
            subject_ids = sample[0].to(device)
            relation_ids = sample[1].to(device)
            object_ids = sample[2].to(device)
            time_ids = sample[3].to(device)
            batch_size = subject_ids.size(0)

            scores = model.predict(subject_ids, relation_ids, time_ids)  # (batch, nrEntities)

            for i in range(batch_size):
                sorted_indices = torch.argsort(scores[i], descending=False)
                target_object = object_ids[i].item()
                rank = (sorted_indices == target_object).nonzero(as_tuple=True)[0].item() + 1

                total_rank += rank
                total_reciprocal_rank += 1.0 / rank
                sample_count += 1

                if rank <= 1:
                    hits1 += 1
                if rank <= 3:
                    hits3 += 1
                if rank <= 10:
                    hits10 += 1

        results = {
            'hits@1': hits1 / sample_count,
            'hits@3': hits3 / sample_count,
            'hits@10': hits10 / sample_count,
            'MR': total_rank / sample_count,
            'MRR': total_reciprocal_rank / sample_count
        }
        print("~~~~Test Results~~~~\n")
        print(f"Hits@1:  {results['hits@1']:.4f}")
        print(f"Hits@3:  {results['hits@3']:.4f}")
        print(f"Hits@10: {results['hits@10']:.4f}")
        print(f"MR:      {results['MR']:.2f}")
        print(f"MRR:     {results['MRR']:.4f}")
