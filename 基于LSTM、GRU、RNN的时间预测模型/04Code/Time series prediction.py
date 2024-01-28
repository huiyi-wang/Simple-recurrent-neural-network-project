import numpy as np
import torch
import pandas as pd
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import warnings
import torch.optim as optimizer
import time
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import os
warnings.filterwarnings("ignore")
'读取数据'
filename = '../00data/1月.xlsx'
dataset = pd.read_excel(filename)
dataset = dataset.values         # 选择原始数据库中需要的数据,并将DataFrame格式转换为ndarray数据类型

'将数据集划分为训练集和测试集'
def split_dataset(dataset,rate_split):
    len = dataset.shape[0]
    train_dataset = dataset[0:int(len * rate_split) , 6:9]
    test_dataset  = dataset[int(len * rate_split):-1 , 6:9]
    return train_dataset,test_dataset

rate_split = 0.7   #原始数据集切分比例
train_dataset,test_dataset = split_dataset(dataset = dataset,rate_split = rate_split)

'对dataset中的数据进行归一化处理,并记录特征和标签的均值和方差'
choice_Normalization = '最大最小值归一化'
feature_X= np.append(train_dataset,test_dataset,axis=0)          #对全部数据（包括训练集和测试集）的特征（X）进行读取
label_Y  = feature_X[:,0]                                        #对全部数据（包括训练集和测试集）的标签（Y）进行读取
label_Y  = label_Y.reshape(-1,1)                                 #进行维度改变
if choice_Normalization == '最大最小值归一化':
    Standardscaler_X = StandardScaler()
    Standardscaler_Y = StandardScaler()
    feature_X = Standardscaler_X.fit_transform(feature_X)
    label_Y   = Standardscaler_Y.fit_transform(label_Y)

    X_mean  = Standardscaler_X.mean_
    X_var   = Standardscaler_X.var_
    X_scale = Standardscaler_X.scale_

    Y_mean  = Standardscaler_Y.mean_
    Y_var   = Standardscaler_Y.var_
    Y_scale = Standardscaler_Y.scale_

elif choice_Normalization == '均值方差归一化' :
    Standardscaler_X = StandardScaler()
    Standardscaler_Y = StandardScaler()
    feature_X = Standardscaler_X.fit_transform(feature_X)
    label_Y   = Standardscaler_Y.fit_transform(label_Y)

    X_mean  = Standardscaler_X.mean_
    X_var   = Standardscaler_X.var_
    X_scale = Standardscaler_X.scale_

    Y_mean  = Standardscaler_Y.mean_
    Y_var   = Standardscaler_Y.var_
    Y_scale = Standardscaler_Y.scale_

'构建时间序列需要的数据集'
def createDataset(dataset, look_back, ahead_step, feature_num):
    '''
    :param dataset: 表示需要进行操作的数据库
    :param look_back: 选择多少个历史数据进行分析
    :param ahead_step: 预测步长为多少
    :param feature_num: 选择多少个特征进行预测
    :return:
    '''
    dataX = []
    dataY = []
    for i in range(len(dataset) - look_back - ahead_step + 1):
        dataX.append(dataset[i : i + look_back])
        dataY.append(dataset[i + (look_back - 1) + ahead_step])
    dataX = torch.tensor(dataX)
    dataX = dataX.reshape(-1 , feature_num)
    dataY = torch.tensor(dataY)
    dataY = dataY[: , 0]
    dataY = dataY.reshape(-1 , 1)
    return dataX, dataY

feature_num = 3                   # 特征个数
ahead_step = 1                    # 预测步数
look_back = 20                    # 使用的历史数据的个数

#对训练集进行归一化
Standardscaler_trainX = StandardScaler()
Standardscaler_trainX.mean_  = X_mean
Standardscaler_trainX.var_   = X_var
Standardscaler_trainX.scale_ = X_scale

#对测试集进行归一化
Standardscaler_testX = StandardScaler()
Standardscaler_testX.mean_  = X_mean
Standardscaler_testX.var_   = X_var
Standardscaler_testX.scale_ = X_scale


train_dataset = Standardscaler_trainX.transform(train_dataset)   #训练集归一化
test_dataset  = Standardscaler_testX.transform(test_dataset)     #测试集归一化
train_dataX, train_dataY = createDataset(dataset = train_dataset,look_back = look_back,ahead_step = ahead_step,feature_num = feature_num)
test_dataX ,  test_dataY = createDataset(dataset = test_dataset ,look_back = look_back,ahead_step = ahead_step,feature_num = feature_num)

'将dataX，dataY转换为LSTM模型的输入格式类型'
train_dataX = train_dataX.reshape(-1,look_back,feature_num)
train_dataY = train_dataY.reshape(-1,1,1)
test_dataX  = test_dataX.reshape (-1,look_back,feature_num)
test_dataY  = test_dataY.reshape (-1,1,1)

'搭建模型LSTM、GRU、RNN'
class LSTMModel(nn.Module):
    def __init__(self, input_Size, hidden_Size, num_layers , dropout , batch_first):
        super(LSTMModel, self).__init__()
        # LSTM层-> 两个LSTM单元叠加
        self.lstm = nn.LSTM(input_size = input_Size,hidden_size = hidden_Size, num_layers = num_layers,dropout = dropout,batch_first = batch_first)
        self.linear = nn.Linear(hidden_Size,1)  # 线性输出
    def forward(self,x):
        lstm_out,hidden = self.lstm(x)
        linear_out=self.linear(lstm_out)
        return linear_out

class RNNModel(nn.Module):
    def __init__(self, input_Size, hidden_Size,num_layers, dropout , batch_first):
        super(RNNModel, self).__init__()
        self.rnn = nn.RNN(input_size = input_Size,hidden_size = hidden_Size, num_layers = num_layers,dropout = dropout,batch_first = batch_first)
        self.linear = nn.Linear(hidden_Size,1)  # 线性输出
    def forward(self,x):
        lstm_out,hidden = self.rnn(x)
        print(hidden.shape)
        linear_out = self.linear(lstm_out)
        return linear_out

class GRUModel(nn.Module):
    def __init__(self, input_Size, hidden_Size,num_layers,dropout , batch_first):
        super(GRUModel, self).__init__()
        self.GRU = nn.GRU(input_size = input_Size,hidden_size = hidden_Size, num_layers = num_layers,dropout = dropout,batch_first = batch_first)
        self.linear = nn.Linear(hidden_Size,1)  # 线性输出
    def forward(self,x):
        lstm_out,hidden = self.GRU(x)
        linear_out = self.linear(lstm_out)
        return linear_out

'网络参数'
dropout = 0.2
batch_first = True
hidden_Size = 30
num_layers = 2

Model_LSTM = LSTMModel(input_Size = feature_num,hidden_Size = hidden_Size,num_layers = num_layers,dropout = dropout,batch_first = batch_first)
Model_LSTM.double() #将所有的浮点类型的参数和缓冲转换为(双浮点)double数据类型.
'进行预测'
# 创建一个一模一样的模型，加载预训练模型的参数
Model_LSTM = LSTMModel(input_Size = feature_num,hidden_Size = hidden_Size,num_layers = num_layers,dropout = dropout,batch_first = batch_first)
Model_LSTM.double() #将所有的浮点类型的参数和缓冲转换为(双浮点)double数据类型.
Model_LSTM.load_state_dict(torch.load("../03Save_Model/Model_LSTM时间序列预测.pt"))
Model_LSTM.eval()  # 不启用 BatchNormalization 和 Dropout，保证BN和dropout不发生变化


pred_testX = Model_LSTM(test_dataX)     # 对测试集进行拟合
pred_trainX = Model_LSTM(train_dataX)   # 对训练集进行拟合
pred_testX  = pred_testX.detach().cpu().numpy()[:,-1,:]  #取LSTM模型的最后一个输出单元 y 的值作为模型的最终预测结果（重点理解）
pred_trainX = pred_trainX.detach().cpu().numpy()[:,-1,:]

Standardscaler_testY = StandardScaler() #对标签Y进行反归一化
Standardscaler_testY.mean_  = Y_mean
Standardscaler_testY.var_   = Y_var
Standardscaler_testY.scale_ = Y_scale

pred_testX  = Standardscaler_testY.inverse_transform(pred_testX)  #测试集的预测输出进行反归一化
pred_trainX = Standardscaler_testY.inverse_transform(pred_trainX) #训练集的预测输出进行反归一化


test_dataY = test_dataY.detach().cpu().numpy()[:,-1,:]
train_dataY = train_dataY.detach().cpu().numpy()[:,-1,:]
test_dataY  = Standardscaler_testY.inverse_transform(test_dataY) #测试集的真实输出进行反归一化
train_dataY = Standardscaler_testY.inverse_transform(train_dataY) #训练集的真实输出进行反归一化

r2_test = r2_score(test_dataY,pred_testX)                          # 计算测试集的相关系数
MSE_test = metrics.mean_squared_error(test_dataY,pred_testX)       # 计算测试集的均方误差
RMSE_test = metrics.mean_squared_error(test_dataY,pred_testX)**0.5 # 计算测试集的均方根误差
print('测试集预测完成')
print('测试集相关系数为R^2为',r2_test)
print('测试集的RMSE为',RMSE_test)
print('测试集的MSE为',MSE_test)

r2_train = r2_score(pred_trainX,train_dataY)                          # 计算训练集的相关系数
MSE_train = metrics.mean_squared_error(pred_trainX,train_dataY)       # 计算训练集的均方误差
RMSE_train = metrics.mean_squared_error(pred_trainX,train_dataY)**0.5 # 计算训练集的均方根误差
print('训练集预测完成')
print('训练集相关系数R^2为',r2_train)
print('训练集的RMSE为',RMSE_train)
print('训练集的MSE为',MSE_train)

'画图'
plt.rcParams['font.sans-serif'] = 'KaiTi'  # 正常显示中文
plt.plot(np.arange(len(test_dataY)), test_dataY,  label="实际值", color='blue', linewidth = 2)  # , label="实际值"
plt.plot(np.arange(len(pred_testX)), pred_testX, label='预测值', linestyle='--', color='red', linewidth = 2)  # , label='L预测值'
plt.legend()
plt.show()
