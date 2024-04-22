from __future__ import print_function
# import os
# import argparse
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
# import torch.optim as optim

from sklearn.metrics import accuracy_score # 用于计算分类任务的准确性 ()

'''
Part of format from pytorch examples repo: 
https://github.com/pytorch/examples/blob/master/mnist/main.py
'''

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.l1 = nn.Linear(3*224*224, 512) #（input, output）
        self.l2 = nn.Linear(512, 128)
        self.l3 = nn.Linear(128, 3)
    def forward(self, x): 
        x = x.view(-1,3*224*224) # change image data to one-dimension vector
        nn = F.relu(self.l1(x))
        nn = F.relu(self.l2(nn))
        nn = self.l3(nn)
        return nn

# 模型训练函数
def train(model, device, train_loader, optimizer):
    model.train()
    cost = nn.CrossEntropyLoss()
    with tqdm(total=len(train_loader)) as progress_bar: # tpdm: progress bar
        for batch_idx, (data, label) in tqdm(enumerate(train_loader)):
            data = data.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = cost(output, label)        
            loss.backward()
            optimizer.step() #update params
            progress_bar.update(1)

# 模型验证函数
def val(model, device, val_loader, checkpoint=None):
    if checkpoint is not None:
        model_state = torch.load(checkpoint)
        model.load_state_dict(model_state)     
    # Need this line for things like dropout etc.  
    model.eval() #pytorch中将模型设置为评估模式的方法，通常用于在测试集上进行模型评估或进行推断预测
    preds = []
    targets = []
    cost = nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad(): #torch.no_grad()上下文管理器， 表示在评估模型时不需要计算梯度
        for batch_idx, (data, label) in enumerate(val_loader): #循环遍历每个批次的数据进行评估
            data = data.to(device)
            target = label.clone() #指验证集中样本的真实标签，也就是每个样本对应的正确答案
            output = model(data)
            # 模型输出和真实标签添加到对应的列表中
            preds.append(output.cpu().numpy())
            targets.append(target.cpu().numpy())
            losses.append(cost(output, label))
    loss = np.mean(losses)
    # 将列表中的预测结果和真实标签进行合并和处理
    '''
        没搞懂这块的 preds 和 targets中的数据
    '''
    preds = np.argmax(np.concatenate(preds), axis=1)
    print('preds',preds)
    targets  = np.concatenate(targets)
    acc = accuracy_score(targets, preds)
    return loss, acc
# 测试函数，不需要cost 和loss
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

