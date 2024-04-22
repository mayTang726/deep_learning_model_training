from __future__ import print_function
import os
import time
import json
# argparse 命令行帮助处理工具
import argparse 

import torch
import torch.nn as nn # 创建模型
import torch.optim as optim #优化工具
import torch.nn.functional as F #
from torch.utils.data import DataLoader # 数据加载器 用于加载和处理数据集 以进行数据批处理
from torchvision import datasets, transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split # split dataset

import models
from utils import Datasets
from utils.params import Params
from utils.plotting import plot_training
from utils import Change_dataset

# learning rate
from torch.optim.lr_scheduler import StepLR 

def main():
    start_time = time.strftime("%d%m%y_%H%M%S")
    # 
    parser = argparse.ArgumentParser()  
    # 添加命令行处理参数，以下是添加了 model_name和--write_data两个参数
    parser.add_argument(
        "model_name", 
        type=str, 
        help="Pass name of model as defined in hparams.yaml."
        )
    parser.add_argument(
        "--write_data",
        required = False,
        default=False,
                help="Set to true to write_data."
        )
    # 解析命令行参数
    args = parser.parse_args()
    # Parse our YAML file which has our model parameters. 
    params = Params("hparams.yaml", args.model_name)
    # params = Params("hparams.yaml", mlp)

    '''
        运行环境需要改动
    '''
    # os.environ 包含环境变量的字典
    # CUDA 并行计算平台和编程模型，CUDA 平台提供了一系列的工具、库和编程接口，使开发者能够在 GPU 上编写高效的并行计算程序
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" # CUDA 设备将按照 PCI 总线 ID 来排序
    os.environ["CUDA_VISIBLE_DEVICES"]=params.gpu_vis_dev # params.gpu_vis_dev = 0 表示第一个cuda设备可见
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # Check if a GPU is available and use it if so. 
    use_gpu= torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")
    
    '''
        运行环境需要改动
    '''


    # Load model that has been chosen via the command line arguments.
    '''
        1. __import__() 接受一个字符串参数，代表导入模块的路径 ，最终的路径为： models.mlp / models.cnn 
        2. fromlist=['object']表示从导入的模块中导入一个名为object的对象，保证
    '''
    model_module = __import__('.'.join(['models', params.model_name]),  fromlist=['object'])
    model = model_module.net()
    # Send the model to the chosen device. 
    # To use multiple GPUs
    # model = nn.DataParallel(model)
    model.to(device)

    # Grap your training and validation functions for your network.
    # 添加训练、验证、优化方法
    train = model_module.train
    val = model_module.val
    test = model_module.test
    optimizer = optim.Adam(model.parameters(), lr=params.lr, weight_decay=0.001) #建议值 0.0005
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
    '''
        pretreatment dataset start
    '''
    data_all = pd.read_csv('wikiArt/wikiart_art_pieces.csv') #get total dataset
    data = data_all[data_all['genre'].isin(['portrait', 'landscape', 'abstract'])]
    data = data[['genre','file_name']] # only use genre and file_name as training dataset content
    data = data.rename(columns = {'genre' : 'label'})
    # use dictionary create number encode relates to each classification
    label_to_code = {}
    current_code = 0
    for label in data['label'].unique():
        label_to_code[label] = current_code
        current_code += 1
    data['label'] = data['label'].map(label_to_code)

    # separate dataset
    X = data.drop('label', axis=1)
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2023)
    X_train, X_val, y_train, y_val = train_test_split(X_train,  y_train, test_size=0.2, random_state=2023)

    # file_path = os.path.join(params.data_dir, 'X_train.csv')
    # if os.path.exists(file_path) == False:
    #     Change_dataset.change_dataset(params, X_train, y_train, X_val, y_val, X_test, y_test)
    
    
    # save dataset to responding folder
    X_train.to_csv(os.path.join(params.data_dir, "X_train.csv"), index=False)  
    X_test.to_csv(os.path.join(params.data_dir, "X_test.csv"), index=False) 
    X_val.to_csv(os.path.join(params.data_dir, "X_val.csv"), index=False) 
    y_train.to_csv(os.path.join(params.data_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(params.data_dir, "y_test.csv"), index=False) 
    y_val.to_csv(os.path.join(params.data_dir, "y_val.csv"), index=False) 
    

    # getattr(模块名, 类名)
    Dataset = getattr(Datasets, params.dataset_class) #从模块Datasets中获取名为params.dataset_class的类
    train_data = Dataset(params.data_dir,"X_train.csv","y_train.csv", flatten=params.flatten) #flatten用于指示是否要对图像数据进行扁平化处理
    val_data = Dataset(params.data_dir,"X_val.csv","y_val.csv", flatten=params.flatten)
    test_data = Dataset(params.data_dir,"X_test.csv","y_test.csv", flatten=params.flatten)


    train_loader = DataLoader(
        train_data, 
        batch_size=params.batch_size, #每个批次中包含的样本数量
        shuffle=True #表示在每个 epoch 开始时是否对数据进行洗牌（即随机打乱数据），这有助于训练时模型更好地泛化
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    test_loader = DataLoader(
        test_data,
        batch_size=params.batch_size,
        shuffle=False
    )
    # 检查创建目录是否存在，不存在的话就创建 logs/模型名 目录
    if not os.path.exists(params.log_dir): os.makedirs(params.log_dir)
    if not os.path.exists(params.checkpoint_dir): os.makedirs(params.checkpoint_dir)
    if not os.path.exists("figs"): os.makedirs("figs") #存放图片文件夹

    #创建数组存放训练之后的 accuracy 和 loss
    val_accs = []
    val_losses = []
    train_losses = []
    train_accs = []
    test_accs = []
    '''
        循环进行模型训练
    '''
    for epoch in range(1, params.num_epochs + 1):
        print("Epoch: {}".format(epoch))
        # Call training function. 
        train(model, device, train_loader, optimizer)
        # Evaluate on both the training and validation set. 
        train_loss, train_acc = val(model, device, train_loader)
        val_loss, val_acc = val(model, device, val_loader)
        test_acc = test(model, device, test_loader)
        # update learning rate
        scheduler.step()
        # Collect some data for logging purposes. 
        train_losses.append(float(train_loss))
        train_accs.append(train_acc)
        val_losses.append(float(val_loss))
        val_accs.append(val_acc)
        test_accs.append(test_acc)
        print('\n\ttrain Loss: {:.6f}\ttrain acc: {:.6f} \n\tval Loss: {:.6f}\tval acc: {:.6f} \n\ttest acc: {:.6f}'.format(train_loss, train_acc, val_loss, val_acc, test_acc))
        
        # Here is a simply plot for monitoring training. 
        # Clear plot each epoch，并将生成图片存入figs文件夹中 
        fig = plot_training(train_losses, train_accs,val_losses, val_accs)
        fig.savefig(os.path.join("figs", "{}_training_vis".format(args.model_name)))

        # Save model every few epochs (or even more often if you have the disk space).
        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(params.checkpoint_dir,"checkpoint_{}_epoch_{}".format(args.model_name,epoch)))
    
    # Some log information to help you keep track of your model information. 
    logs ={
        "model": args.model_name,
        "train_losses": train_losses,
        "train_accs": train_accs,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "test_accs": test_accs,
        "best_val_epoch": int(np.argmax(val_accs)+1),
        "model": args.model_name,
        "lr": params.lr,
        "batch_size":params.batch_size
    }

    with open(os.path.join(params.log_dir,"{}_{}.json".format(args.model_name,  start_time)), 'w') as f: # 'w' 表示以写入的模式打开，如果文件不存在直接创建文件，如果文件已存在则先清空文件内容再写入文件
        json.dump(logs, f)



if __name__ == '__main__':
    main()
