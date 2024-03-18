import torch
import torch.nn as nn

from diagonal.data import get_classification


class DeepClassifier(nn.Module):
    def __init__(self, in_features=2, out_features=40, factor=1):
        super(DeepClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.fc1 = nn.Linear(in_features=in_features, out_features=out_features)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight / factor)

        self.fc2 = nn.Linear(in_features=out_features, out_features=out_features)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight / factor)

        self.fc3 = nn.Linear(in_features=out_features, out_features=out_features)
        self.fc3.weight = torch.nn.Parameter(self.fc3.weight / factor)

        self.fc4 = nn.Linear(in_features=out_features, out_features=out_features)
        self.fc4.weight = torch.nn.Parameter(self.fc4.weight / factor)

        self.fc_final = nn.Linear(in_features=out_features, out_features=1)
        self.fc_final.weight = torch.nn.Parameter(self.fc_final.weight / factor)

    def forward(self, x):
        res = torch.relu(self.fc1(x))
        res = torch.relu(self.fc2(res))
        res = torch.relu(self.fc3(res))
        res = torch.relu(self.fc4(res))
        res = torch.sigmoid(self.fc_final(res))
        return res

    def forward_no_sigmoid(self, x):
        with torch.no_grad():
            res = torch.relu(self.fc1(x))
            res = torch.relu(self.fc2(res))
            res = torch.relu(self.fc3(res))
            res = torch.relu(self.fc4(res))
            return self.fc_final(res)

    def get_margin(self, dataset):
        dists = torch.zeros(dataset.shape[0])
        for i in range(dataset.shape[0]):
            classification = get_classification(dataset[i, :].unsqueeze(0)).item()
            score = self.forward_no_sigmoid(dataset[i, :])
            dists[i] = ((classification - 0.5) * 2) * score
        return torch.min(dists)







