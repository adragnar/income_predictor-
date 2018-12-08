import torch.utils.data as data


class AdultDataset(data.Dataset):

    def __init__(self, feature_values, labels):  #feature_values 2D numpy array of featurevals per sample, labels are classifications
        ######

        # 3.1 YOUR CODE HERE
        self.features = feature_values
        self.labels = labels
        pass
        ######

    def __len__(self):  #return number of different people that are in dataset
        return len(self.features)

    def __getitem__(self, index):  #return all feature info relates to ith person in dataset
        ######

        # 3.1 YOUR CODE HERE
        return self.features[index], self.labels[index]
        ######