import numpy as np
from torch.utils.data.dataset import Dataset

def categorize(labels, dict_labels=None):
    possible_labels = list(set(labels))

    if not dict_labels:
        dict_labels = {}
        n = 0
        for label in possible_labels:
            dict_labels[label] = n
            n = n + 1

    new_labels = []
    for label in labels:
        new_labels.append(dict_labels[label])

    return new_labels


def load_data(path):
    """Load and shape the dataset"""
    npzfile = np.load(path)
    data = npzfile["data"]
    labels = npzfile["labels"]
    npzfile.close()

    data = data.reshape(data.shape[0], data.shape[1], 1)
    labels = categorize(labels)       

    return data, labels


class WFDataset(Dataset):
    def __init__(self, path):
        self.npzfile = np.load(path)
        self.raw_data = self.npzfile["data"].astype(dtype="float_")
        self.data = np.transpose(self.raw_data.reshape(self.raw_data.shape[0], self.raw_data.shape[1], 1), (0, 2, 1))
        self.label = categorize(self.npzfile["labels"])
        self.data_len = len(self.data)
        self.npzfile.close()
        
    def __getitem__(self, index):
        single_label = self.label[index]
        return (self.data[index], single_label)
 
    def __len__(self):
        return self.data_len



if __name__ == "__main__":
    data = trainDataset("tor_100w_2500tr.npz")