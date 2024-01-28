import pickle
import numpy as np
import torch
import os
from gensim.models.word2vec import Word2Vec
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import nn
from torch import optim

def split_data(file_path = '../00data/poetry_7.txt'):
    all_data = open(file_path,'r',encoding = 'utf_8').read()
    all_data_split = " ".join(all_data)
    with open ("../01Createdataset/split.txt",'w',encoding = "utf_8") as f:
        f.write(all_data_split)

def train_ver(split_path = '../01Createdataset/split.txt', org_file = '../00data/poetry_7.txt'):
    # 如果从split_path这个路径没有发现split.txt的话，就直接调用split_data函数生成一个split.txt文件
    if os.path.exists(split_path) == False:
        split_data()
    split_all_data = open(split_path,'r',encoding = 'utf_8').read().split('\n')        # 转换数据类型，将字符串类型转换为list类型，方便Word2Vec模型对其进行编码
    org_data = open(org_file,'r',encoding = 'utf_8').read().split('\n')                # 转换数据类型，将字符串类型转换为list类型
    model = Word2Vec(split_all_data,vector_size = 128 ,min_count = 1, workers = 6)     # vertor_size为自己定义赋给的值，表示用107维的向量来表示一个字
    return org_data, (model.syn1neg ,model.wv.key_to_index , model.wv.index_to_key)

def generate_poetry():
    result = ""
    word_index = np.random.randint(0,word_size,1)[0]
    result = result + index_2_word[word_index]
    for i in range(31):
        word_embedding = w1[word_index]
        pre = net(torch.tensor(word_embedding).view(1,1,-1))
        word_index = int(torch.argmax(pre))
        result = result + index_2_word[word_index]
    print(result)

class Mydataset(Dataset):
    #加载所有的数据，并存储和初始化一些数据集
    def __init__(self,all_data,w1,word_2_index,index_2_word):
        self.all_data = all_data          # 原始词库
        self.w1 = w1                      # 编码后的词库
        self.word_2_index = word_2_index
        self.index_2_word = index_2_word

    #获取一条数据，并进行封装
    def __getitem__(self,index):
        a_poetry_words = self.all_data[index]
        a_poetry_index = [self.word_2_index[word] for word in a_poetry_words]
        x_index = a_poetry_index[:-1]       # x 的词库对应索引
        y_index = a_poetry_index[1:]        # y 的词库对应索引
        x_embodding = self.w1[x_index]      # x 的对应的词向量
        return x_embodding,torch.tensor(y_index)

    def __len__(self):
        lenge = len(all_data)
        return lenge


class Mymodel(nn.Module):
    def __init__(self,input_size,hidden_size,word_size,num_layer):
        'input_size 表示 输入的特征数量 ，此处表示使用词向量大小，此处为 128 '
        'hidden_size 表示 隐藏单元的个数'
        'word_size 表示 词库的大小 '
        super(Mymodel, self).__init__()
        self.lstm = nn.LSTM(input_size =input_size,hidden_size = hidden_size,num_layers = num_layer,batch_first = True, bidirectional = False)
        self.flatten = nn.Flatten(0,1)                # 将第 0 维 和 第 1 维 合成一个维度
        self.linear = nn.Linear(hidden_size,word_size)
        self.dropout = nn.Dropout(0.3)

    def forward(self,x):
        hidden,(h,c)= self.lstm(x)
        hidden_dropout = self.dropout(hidden)
        flatten_out = self.flatten(hidden_dropout)
        pre = self.linear(flatten_out)
        return pre

if __name__ == "__main__":

    '参数定义'
    epoch = 100
    batch_size = 128
    all_data , ( w1,word_2_index,index_2_word ) = train_ver()

    '加载数据'
    dataset = Mydataset(all_data = all_data ,w1 = w1, word_2_index = word_2_index, index_2_word = index_2_word)
    dataloader = DataLoader(dataset = dataset ,batch_size = batch_size,shuffle = True,num_workers = 2)


    hidden_size = 50
    learn_rate = 0.001
    num_layer = 2
    word_size = w1.shape[0]   # 表示整个词集的大小
    input_size = w1.shape[1]  # 表示词向量大小 == 模型的输入input的维度
    net = Mymodel(input_size = input_size,hidden_size = hidden_size,word_size = word_size,num_layer = num_layer)
    optimizer = optim.Adam(net.parameters(),lr = learn_rate)
    Loss = nn.CrossEntropyLoss()

    '训练过程'
    for num_epoch in range(epoch):
        for i,data in enumerate(dataloader):
            x , y = data
            pre_y = net(x)
            loss = Loss( pre_y, y.reshape(-1) )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i % 10 == 0:
                print('num_epoch:' + str(num_epoch),'i:' + str(i),'loss:' + str(loss.data))
                generate_poetry()

    torch.save(net, '../03Save_model/基于LSTM的古诗生成.pt')



