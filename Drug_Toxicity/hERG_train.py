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

torch.manual_seed(2021)
np.random.seed(2020)

train_feature = pickle.load(open('./Data_file/train_feature.pkl','rb'))
train_label = pickle.load(open('./Data_file/ADMET_label.pkl','rb'))
train_feature = torch.tensor(train_feature)
train_label = torch.tensor(train_label)

def best_f_1(label,output):
    f_1_max = 0
    t_max = 0
    for t in range(1,100):
        threshold = t / 100
        predict = np.where(output>threshold,1,0)
        f_1 = skm.f1_score(label, predict, pos_label=0)
        if f_1 > f_1_max:
            f_1_max = f_1
            t_max = threshold

    pred = np.where(output>t_max,1,0)
    accuracy = skm.accuracy_score(label, pred)
    recall = skm.recall_score(label, pred)
    precision = skm.precision_score(label, pred)
    MCC = skm.matthews_corrcoef(label, pred)
    return accuracy,recall,precision,MCC,f_1_max,t_max

class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Sequential()
        self.fc1.add_module('fc1', nn.Linear(729, 128))
        self.fc1.add_module('ReLU1', nn.ReLU())

        self.dropout = nn.Dropout(0.2)
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

def train_epoch(model,train_loader, optimizer, loss_fun):
    model.train()
    loss = 0
    num = 0
    for step, (feature, label) in enumerate(train_loader):
        feature_var = torch.autograd.Variable(feature.float().to(DEVICE))
        label_var = torch.autograd.Variable(label.float().to(DEVICE))

        optimizer.zero_grad()
        output = model(feature_var)
        train_loss = loss_fun(output, label_var).to(DEVICE)
        train_loss.backward()
        optimizer.step()
        loss = loss + train_loss.item()
        num = num + len(label)
    epoch_loss = loss / num
    return epoch_loss

def valid_epoch(model, valid_loader, loss_fun):
    model.eval()
    predict = []
    v_label = []
    loss = 0
    num = 0
    with torch.no_grad():
        for step, (feature, label) in enumerate(valid_loader):
            feature_var = torch.autograd.Variable(feature.float().to(DEVICE))
            label_var = torch.autograd.Variable(label.float().to(DEVICE))

            pred = model(feature_var)
            valid_loss = loss_fun(pred, label_var).to(DEVICE)
            loss = loss + valid_loss.item()
            num = num + len(label)
            predict.extend(pred.data.cpu().numpy())
            v_label.extend(label.data.cpu().numpy())
        accuracy, recall, precision, MCC, f_1, t_max, = best_f_1(np.array(v_label), np.array(predict))
    epoch_loss = loss / num
    return epoch_loss,accuracy, recall, precision, f_1, t_max

def train(model, train_feature = train_feature, train_label = train_label, mode =0 ):
    label = train_label[:,mode]
    feature_label = torch.utils.data.TensorDataset(train_feature,label)
    samples_num = len(feature_label)
    split_num = int(0.8 * samples_num)
    data_index = np.arange(samples_num)
    np.random.shuffle(data_index)
    train_index = data_index[:split_num]
    valid_index = data_index[split_num:]
    train_sampler = SubsetRandomSampler(train_index)
    valid_sampler = SubsetRandomSampler(valid_index)
    if torch.cuda.is_available():
        train_loader = torch.utils.data.DataLoader(dataset=feature_label, batch_size=BATCH, sampler=train_sampler,
                                                   drop_last=False, num_workers=5, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(dataset=feature_label, batch_size=BATCH, sampler=train_sampler,
                                                   drop_last=False)

    if torch.cuda.is_available():
        valid_loader = torch.utils.data.DataLoader(dataset=feature_label, batch_size=BATCH,
                                                   sampler=valid_sampler, drop_last=False, num_workers=5,
                                                   shuffle=False, pin_memory=True)
    else:
        valid_loader = torch.utils.data.DataLoader(dataset=feature_label, batch_size=BATCH, sampler=valid_sampler,
                                                   shuffle=False)
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=LEARN_RATE)
    loss_fun = F.binary_cross_entropy
    train_log = []
    valid_log = []
    valid_f_1 = []
    valid_acc = []
    max_f_1 = 0
    max_acc = 0
    max_epoch = 0
    Threshold = 0
    for epoch in range(EPOCH):
        start_time = time.time()
        loss = train_epoch(model, train_loader, optimizer, loss_fun)
        train_log.append(loss)
        loss_v, accuracy, recall, precision,f_1, t_max = valid_epoch(model, valid_loader, loss_fun)
        end_time = time.time()
        valid_log.append(loss_v)
        valid_f_1.append(f_1)
        valid_acc.append(accuracy)
        if f_1 > max_f_1:
            max_f_1 = f_1
            max_acc = accuracy
            max_epoch = epoch
            Threshold = t_max
            torch.save(model.state_dict(), './hERG/best_model.dat')
        print("Epoch: ", epoch + 1, "|", "Epoch Time: ", end_time - start_time, "s")
        print("Train loss: ", loss)
        print("valid loss: ", loss_v)
        print("accuracy:" , accuracy)
        print("precision:", precision)
        print("recall:" ,recall)
        print("F_1_score: ",f_1)
        print("Threshold: ",t_max)

    plt.plot(train_log, 'r-')
    plt.title('train_loss')
    plt.savefig("./hERG/train_loss.png")
    plt.close()

    plt.plot(valid_log, 'r-')
    plt.title('valid_loss')
    plt.savefig("./hERG/valid_loss.png")
    plt.close()

    plt.plot(valid_f_1, 'r-')
    plt.title('valid_f_1')
    plt.savefig("./hERG/valid_f_1.png")
    plt.close()

    plt.plot(valid_acc, 'r-')
    plt.title('valid_acc')
    plt.savefig("./hERG/valid_acc.png")
    plt.close()
    return max_f_1, max_acc, max_epoch, Threshold

if __name__ == '__main__':
    path_dir = "./hERG"
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)
model = MLP().to(DEVICE)
f_1, acc, epoch ,Threshold = train(model,train_feature,train_label,2)
print("Epoch: ", epoch )
print("accuracy:" , acc)
print("F_1_score: ",f_1)
print("Threshold: ",Threshold)
f = open('./hERG/hERG.txt','w')
f.write("Epoch: "+ str(epoch)  + '\n')
f.write("accuracy:" + str(acc) + '\n')
f.write("F_1_score: " + str(f_1) + '\n')
f.write("Threshold: " + str(Threshold) + '\n')

