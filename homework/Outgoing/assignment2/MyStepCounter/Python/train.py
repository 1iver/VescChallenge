# %%
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from cmath import pi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os
# %%
class HARNetwork1d(nn.Module):
    def __init__(self,input_channel, channel_length):  # channel 3 or 4, large len
        super(HARNetwork1d, self).__init__()
        self.channel_length = channel_length
        self.conv1 = nn.Conv1d(in_channels=input_channel,out_channels=64,kernel_size=3,padding=1)
        self.activation1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1)
        self.activation2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.conv3 = nn.Conv1d(in_channels=128,out_channels=32,kernel_size=3,stride=1,padding=1)
        self.activation3 = nn.ReLU()
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        self.fc1 = nn.Linear(32*(self.channel_length//8), 32)
        self.activation3 = nn.ReLU()
        self.fc2 = nn.Linear(32, 5)
        self.softmax = nn.Softmax(dim=1)
    def forward(self,x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.activation3(x)
        x = self.pool3(x)
        x = x.view(-1, 32*(self.channel_length//8))
        x = self.fc1(x)
        x = self.activation3(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
# %%
class NetDataset(torch.utils.data.Dataset):

    def __init__(self,datas,labels):
        self.datas = datas
        self.labels = labels
    def __len__(self):
        return len(self.datas)
    def __getitem__(self, item):
        return self.datas[item],self.labels[item]
# %%
def slice_data(data, slice_len, label):
    data_len = data.shape[0]
    slice_num = data_len // slice_len
    data = data[:slice_num*slice_len, :]
    data = data.reshape(slice_num, slice_len,3)
    data = np.swapaxes(data,1,2)
    data = (data-np.mean(data,axis=2,keepdims=True))/np.std(data,axis=2,keepdims=True)
    labels = [label]*slice_num
    return data,labels
def load_data(data_dirs, slice_len):
    datas = []
    labels = []
    for data_dir in data_dirs:
        data_dir_path = "MyStepCounter/acc_data/"+data_dir
        for file in os.listdir(data_dir_path):
            if file.endswith('.txt'):
                data = np.loadtxt(data_dir_path + '/' + file)
                reshape_data = data.reshape(-1,3)[150:-150,:]
                data,label = slice_data(reshape_data, slice_len, data_dir)
                datas.append(data)
                labels.append(label)
    datas = np.concatenate(datas, axis=0)
    labels = np.concatenate(labels,axis=0)
    # labels[labels!="walk"]="ELSE"
    labels = LabelEncoder().fit_transform(labels)
    return datas,labels
# %%
datas,labels = load_data(["walk","trot","wave","squats","sit"],25*5)
train_datas,test_datas,train_labels,test_labels = train_test_split(datas,labels,test_size=0.2,stratify=labels,random_state=123)
# %%
train_dataset = NetDataset(train_datas,train_labels)
test_dataset = NetDataset(test_datas,test_labels)
# %%
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True)
# %%
epochs = 100
device = torch.device("cpu")
model = HARNetwork1d(3,25*5).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# %%
train_acc = []
test_acc = []
for epoch in range(epochs):
    model.train()
    train_count = 0
    for i,(datas,labels) in enumerate(train_loader):
        datas = datas.float().to(device)
        labels = labels.to(device)
        outputs = model(datas)
        loss = criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("epoch:",epoch,"step:",i,"loss:",loss.item(),"acc",(outputs.argmax(dim=1)==labels).sum().item()/labels.shape[0])
        train_count += (outputs.argmax(dim=1)==labels).sum().item()
    train_acc.append(train_count/train_datas.shape[0])
    model.eval()
    output = model(torch.from_numpy(test_datas).float())
    print("test_acc", (output.argmax(dim=1) == torch.from_numpy(test_labels)).sum().item() / test_labels.shape[0])
    test_acc.append((output.argmax(dim=1) == torch.from_numpy(test_labels)).sum().item() / test_labels.shape[0])
# %%
torch.save(model.state_dict(),"MyStepCounter/Python/model.pth")
# %%
plt.plot(train_acc,"r",label="train")
plt.plot(test_acc,"y",label="test")
plt.legend()
# %%
torch.load("MyStepCounter/Python/model.pth")
