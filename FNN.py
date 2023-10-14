import math
import torch
import re
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from matplotlib.ticker import FuncFormatter
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
torch.multiprocessing.set_sharing_strategy('file_system')
class Traindataset(Dataset):
    def __init__(self, data_root='train1.dat'):
        super(Traindataset, self).__init__()
        self.frame = []
        with open(data_root, 'r') as f:
            for line in f:
                values = [float(value) for value in filter(None,line.split(" "))]
                self.frame.append(values)
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        result = self.frame[idx]
        return torch.Tensor(result[:-1]), result[-1]
class Testdataset(Dataset):
    def __init__(self, data_root='test.dat'):
        super(Testdataset, self).__init__()
        self.frame = []
        with open(data_root, 'r') as f:
            for line in f:
                values = [float(value) for value in filter(None,line.split(" "))]
                self.frame.append(values)
    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        result = self.frame[idx]
        return torch.Tensor(result[:-1]), result[-1]
class EarlyStopping():
    """
    Early stopping to stop the training when the loss does not improve after
    certain epochs.
    """
    def __init__(self, patience=5, min_delta=0):
        """
        :param patience: how many epochs to wait before stopping when loss is
               not improving
        :param min_delta: minimum difference between new loss and old loss for
               new loss to be considered as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
    def __call__(self, val_loss):
        if self.best_loss == None:
            self.best_loss = val_loss
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            # reset counter if validation loss improves
            self.counter = 0
        elif self.best_loss - val_loss < self.min_delta:
            self.counter += 1
            # print(f"INFO: Early stopping counter {self.counter} of {self.patience}")
            if self.counter >= self.patience:
                print('INFO: Early stopping')
                self.early_stop = True
early = EarlyStopping(patience=15, min_delta=0)
def create_model(num_of_hidden,in_size,dropout):
    layers = []
    layers.append(nn.Linear(7, in_size))
    for i in range(num_of_hidden):
        layers.append(nn.Linear(in_size, in_size))
        layers.append(nn.BatchNorm1d(in_size, momentum=0.08))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(in_size, 1))
    return nn.Sequential(*layers)
class CustomNN(torch.nn.Module):
        def __init__(self, input_shape=8, output_shape=1):
            super().__init__()
            self.layer = torch.nn.Sequential(
                torch.nn.Linear(input_shape, 10),
                # torch.nn.BatchNorm1d(10, momentum=0.5),
                torch.nn.ReLU(),
                # torch.nn.Dropout(0.34),
                torch.nn.Linear(10, 10),
                # torch.nn.BatchNorm1d(10, momentum=0.5),
                torch.nn.ReLU(),
                # torch.nn.Dropout(0.3),
                torch.nn.Linear(10, 10),
                # torch.nn.BatchNorm1d(10, momentum=0.5),
                torch.nn.ReLU(),
                # torch.nn.Dropout(0.34),
                torch.nn.Linear(10, 10),
                # torch.nn.BatchNorm1d(10, momentum=0.5),
                torch.nn.ReLU(),
                # torch.nn.Dropout(0.34),
                torch.nn.Linear(10, 10),
                # torch.nn.BatchNorm1d(10, momentum=0.5),
                torch.nn.ReLU(),
                # torch.nn.Dropout(0.34),
                torch.nn.Linear(10,output_shape)
            )
        def forward(self, input):
                return self.layer(input)
kfold_rmse_train=[]
kfold_rmse_val=[]
kfold_rmse_pre1=[]
kfold_rmse_pre2=[]
data_plot = []
list_result = {}
for m in range(0,5,1):
    list_temp ={}

    with open('reactions1_11_final.pkl','rb') as f:
        list_reaction = pickle.load(f)
    df = pd.concat(list_reaction[1][:-1],axis=0)
    df_1 = df[(df['T'] >= 700) & (df['T'] < 1000)]
    df_2 = df[(df['T'] >= 1000) & (df['T'] < 1500)]
    df_3 = df[(df['T'] >= 1500) & (df['T'] <= 2000)]
    for i in range(10):
        df_1_list = [df_1.sample(frac=0.1) for _ in range(10)]
        df_2_list = [df_2.sample(frac=0.1) for _ in range(10)]
        df_3_list = [df_3.sample(frac=0.1) for _ in range(10)]
        df_train_list = []
        df_test_list = []
        for j in range(10):
            df_train_list.extend([df_1_list[j].iloc[:9],
                                  df_2_list[j].iloc[:9],
                                  df_3_list[j].iloc[:9]])
            df_test_list.extend([df_1_list[j].iloc[9:],
                                 df_2_list[j].iloc[9:],
                                 df_3_list[j].iloc[9:]])
        df_train = pd.concat(df_train_list)
        df_test = pd.concat(df_test_list)
        df_test.to_csv('FNN_cross/{}.dat'.format(str(i+1)),header=None, sep=' ',index=False)
        df_train.to_csv('FNN_cross/{}.dat'.format(str(i+1)+'-'+str(i+1)),header=None,sep=' ',index=False)
    for idx in range(0,10,1):
        # list_temp1 = {}
        print(' Kfold'+str(idx+1)+'开始')
        train_dataset = Traindataset(data_root='FNN_cross/{}.dat'.format(str(idx+1)))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
        val_datase = Testdataset(data_root='FNN_cross/{}.dat'.format(str(idx+1)+'-'+str(idx+1)))
        val_loader = DataLoader(val_datase, batch_size=1, shuffle=True, num_workers=0)
        result=create_model(3,14,0.2)
        net = CustomNN(7, 1)
        net.layer=result
        print(net)
        nepoch=200
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        torch.optim.lr_scheduler.StepLR(optimizer,30,gamma=0.8,last_epoch=-1)
        loss_function = torch.nn.MSELoss(reduction='mean')
        epoch_num=[]
        y_train_loss=[]
        y_test_loss=[]
        y_val_loss=[]
        for epoch in range(nepoch):
            avg_loss_train = 0
            prediction_y_train = []
            x_all_train = []
            y_all_train = []
            for itx, data in enumerate(train_loader):
                x, y = data
                y = y.float().view(-1,1)
                # clear the gradients caculated before
                net.zero_grad()
                # forward pass
                pred_y_train = net(x)
                # compute the loss
                loss = loss_function(pred_y_train, y)
                # backward the loss to compute new gradients
                loss.backward()
                # use optimizer to apply the gradients on the weights in network
                optimizer.step()
                avg_loss_train += loss.item()
                x_all_train.append(x[:,-1])
                y_all_train.append(y[:])
                # prediction_y_train.append(pred_y[:])
                y_hat_train = pred_y_train.data.detach()
                prediction_y_train.append(y_hat_train)
            net.eval()
            #val dataset
            avg_loss_val = 0
            prediction_y_val = []
            for itx, data in enumerate(val_loader):
               x, y = data
               y = y.float().view(-1,1)
               # forward pass
               pred_y = net(x)
               # compute the loss
               test_loss = loss_function(pred_y, y)
               avg_loss_val += test_loss.item()
            avg_loss_train /= len(train_loader)
            rmse_avg_loss_train=np.sqrt(avg_loss_train)
            avg_loss_val /= len(val_loader)
            early.__call__(avg_loss_val)
            rmse_avg_loss_val=np.sqrt(avg_loss_val)
            ####################################################
            ####################################################
            print("epoch: %d   loss: %.5f    valloss: %.5f   count: %.5f "%(epoch, rmse_avg_loss_train,rmse_avg_loss_val,early.counter))
            epoch_num.append(epoch)
            y_train_loss.append(rmse_avg_loss_train)
            y_val_loss.append(rmse_avg_loss_val)
            if epoch==200:
                early.counter=0
                early.early_stop=False
                early.best_loss = None
                print('reaction',idx+1,'已经完成')
                print("epoch: %d   loss: %.5f   valloss: %.5f   count: %.5f "%(epoch, rmse_avg_loss_train,rmse_avg_loss_val,early.counter))
                print('***************************************************************')
                kfold_rmse_train.append(rmse_avg_loss_train)
                kfold_rmse_val.append(rmse_avg_loss_val)
                break
            else:
                if early.early_stop==True:
                    early.counter=0
                    early.best_loss = None
                    early.early_stop=False
        print(' Kfold'+str(idx+1)+'结束')
        print('***************************************************************')