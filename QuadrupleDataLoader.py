import torch
import pandas as pd
from torch.utils.data import Dataset


class ICEWSData(Dataset):
    def __init__(self, subset, fourteen):
        self.subset = subset
        if fourteen:
            self.baseDataPath = "icews14/"
        else:
            self.baseDataPath = "icews05-15/"

        df = pd.read_csv(f'{self.baseDataPath}{subset}.csv', sep=",", header=None)
        self.data = torch.from_numpy(df.values).long()

        print(f"Loaded {len(self.data)} {subset} samples from {self.baseDataPath}")

    def __getitem__(self, index):
        row = self.data[index]
        return row[0], row[1], row[2], row[3]

    def __len__(self):
        return len(self.data)


if __name__ == "__main__":
    dataset = ICEWSData('test', True)
    print(dataset[5])
    print(len(dataset))
