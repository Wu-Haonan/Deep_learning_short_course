import torch
import torch.nn as nn
import pickle
import torch.utils.data
import  matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import torch.optim as optim
import time
import sklearn.metrics as skm
import torch.nn.functional as F
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)
BATCH = 4
EPOCH = 100
LEARN_RATE = 0.0001

torch.manual_seed(2020)
np.random.seed(2020)

train_feature = pickle.load(open('./Data_file/test_feature.pkl','rb'))
train_feature = torch.tensor(train_feature)

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential()
        self.fc1.add_module('fc1', nn.Linear(729, 128))
        self.fc1.add_module('ReLU1', nn.ReLU())

        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Sequential()
        self.fc2.add_module('fc2', nn.Linear(128, 64))
        self.fc2.add_module('ReLU2', nn.ReLU())
        self.fc3 = nn.Sequential(nn.Linear(64, 1), nn.Sigmoid())
    def forward(self,x):
        out = self.fc1(x)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = torch.squeeze(out,dim=1)
        return out

def test(model,test_loader):
    predict = []
    model.eval()
    with torch.no_grad():
        for step, feature in enumerate(test_loader):
            feature_var = torch.autograd.Variable(feature.to(DEVICE).float())
            pred = model(feature_var)
            predict.extend(pred.cpu().data.numpy())

    pred = np.where(np.array(predict)>0.73,1,0)

    f = open('./hERG/hERG_test.txt', 'w')
    f.write(str(pred))
    f.close()

if torch.cuda.is_available():
    test_loader = torch.utils.data.DataLoader(dataset=train_feature, batch_size=BATCH,
                                              num_workers=5,pin_memory=True,shuffle=False)
else:
    test_loader = torch.utils.data.DataLoader(dataset=train_feature, batch_size=BATCH,shuffle=False)

best_model = MLP().to(DEVICE)
best_model.load_state_dict(torch.load('./hERG/best_model.dat'))

test(best_model,test_loader)


