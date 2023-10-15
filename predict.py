import torch
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import re
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.pyplot import MultipleLocator
df = pd.read_csv(r'example.csv',delimiter=',')
print(df)
class CustomNN(torch.nn.Module):
        def __init__(self, input_shape=7, output_shape=1):
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
                torch.nn.Linear(10,output_shape)
            )
        def forward(self, input):
                return self.layer(input)
net = CustomNN()
list_model  = os.listdir(r'Selecting_model_weight')
# list_model =['3-8']
list3=[]
list_std=[]
list4 = []
resultstd=[]
first_row = df.iloc[0, :].tolist()
for i in range(700,2000,20):
    x1=first_row
    x2=[[g]for g in x1]
    x2.append([i])
    x2=np.array(x2).T
    y=torch.FloatTensor(x2)
    print(y)
    result = []
    for g in list_model:
        net = torch.load(r'Selecting_model_weight\{}'.format(g))
        print(net)
        result.append(net(y).tolist()[0])
    # print(result)
    list4.append(np.mean(result))
    resultstd.append(np.std(result))
list3.append(list4)
list_std.append(resultstd)
T_list = [1000/i for i in range(700,2000,20)]
# plt.figure(figsize=(20, 5), dpi=500)
fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(10, 10))
axs.plot(T_list, np.array(list3[0]), color='red', label='predict', marker='o')
axs.set_xlabel('1000/T', fontsize=22,fontweight='bold')
axs.set_ylabel('lg(k)', fontsize=22,fontweight='bold')
fig.suptitle('Predicted Values ', fontweight='bold',fontsize=22)
plt.show()
print('T K')
for i in zip(list(range(700,2000,20)),list3[0]):
    print(i[0],i[1])