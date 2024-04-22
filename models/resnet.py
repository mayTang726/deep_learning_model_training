from __future__ import print_function
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from torchvision import models
'''Part of format and full model from pytorch examples repo: https://github.com/pytorch/examples/blob/master/mnist/main.py'''
class net(nn.Module):
    # input_height, input_width = 224, 224
    # input_channels = 3
    def __init__(self):
        super(net, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        # 修改ResNet的输入通道数
        self.resnet18.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # 修改最后的全连接层
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 3)

    def forward(self, x):
        return self.resnet18(x)

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

