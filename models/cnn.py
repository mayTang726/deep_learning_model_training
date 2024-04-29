from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
'''Part of format and full model from pytorch examples repo: https://github.com/pytorch/examples/blob/master/mnist/main.py'''
class net(nn.Module):
    # input_height, input_width = 224, 224
    # input_channels = 3
    def __init__(self):
        super(net, self).__init__()
        # input_kernel: 3，output_kernel: 32，kernel_size: 3x3，stride: 1, padding:1
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1) # [32, 224, 224]
        self.conv2 = nn.Conv2d(64, 128, 3, 1, 1) #after maxpooling -> [64, 112, 112]
        self.dropout1 = nn.Dropout(0.3)
        self.conv3 = nn.Conv2d(128, 256, 3, 1, 1)  #[128, 56, 56]
        self.dropout2 = nn.Dropout(0.3)
        self.fc1 = nn.Linear(256 * 28 * 28, 128) # after maxpooling ->[64, 28, 28]
        self.dropout3 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 3) # 定义了第二个全连接层，输入大小为上一层的输出大小（即 128），输出大小为类别数量3
    def forward(self, x):
        nn = self.conv1(x) # 经过第一个卷积层
        nn = F.relu(nn) # 经过 ReLU 激活函数
        nn = F.max_pool2d(nn, 2) #[2,2,0]

        nn = self.conv2(nn) # 经过第二个卷积层，
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2) # [2,2,0]
        nn = self.dropout1(nn) # 应用第一个dropout层

        nn = self.conv3(nn) # 经过第二个卷积层，
        nn = F.relu(nn) 
        nn = F.max_pool2d(nn, 2) # [2,2,0]
        nn = self.dropout2(nn) # 应用第三个dropout层

        nn = torch.flatten(nn, 1) # 将特征图平为一维向量，用于linear层
        nn = self.fc1(nn) # 第一个全连接层
        nn = F.relu(nn) # 经过 ReLU 激活函数
        nn = self.dropout3(nn) # 再应用第二个 Dropout 层
        output = self.fc2(nn) # 经过第二个全连接层

        return output 

def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar:
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)        
            loss.backward()
            optimizer.step()
            progress_bar.update(1)



def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)     
    # Need this line for things like dropout etc.  
    model.eval()
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(val_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label))
    loss = np.mean(losses)
    preds = np.argmax(np.concatenate(preds), axis=1)
    print('preds',preds)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc

def test(model, device, test_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)    
    model.eval()
    preds = []
    targets = []
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            data = data.to(device)
            target = label.clone()
            output = model(data)
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
        
    preds = np.argmax(np.concatenate(preds), axis=1)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return acc

