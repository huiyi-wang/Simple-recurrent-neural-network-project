import os
import torch
import time
import pickle
import numpy as np
import pandas as pd
import torch.nn as nn
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def Create_parameter_data(file_path = '../00data/translate.csv'):

    if (os.path.exists('../01Create_parameter_data/中文（原文-索引）.pkl') and os.path.exists('../01Create_parameter_data/英文（原文-索引）.pkl')) == False:
        data = pd.read_csv(file_path)
        English_data = data.iloc[:,[0]]
        Chinese_data = data.iloc[:,[1]]
        English_data = np.array(English_data).reshape(-1)  # 将 Dataframe格式转换为np.array格式，并进行降维
        Chinese_data = np.array(Chinese_data).reshape(-1)  # 将 Dataframe格式转换为np.array格式，并进行降维
        English_data = English_data.tolist()   # 再将np.array格式转换为list格式
        Chinese_data = Chinese_data.tolist()   # 再将np.array格式转换为list格式
        English_data = str(English_data)       # 将list格式转换为str格式
        Chinese_data = str(Chinese_data)       # 将list格式转换为str格式
        if (os.path.exists('../01Create_parameter_data/English_data.txt') and os.path.exists('../01Create_parameter_data/Chinese_data.txt')) == False:
            with open('../01Create_parameter_data/English_data.txt','w',encoding='utf-8') as Write:
                Write.write(str(English_data))
            with open('../01Create_parameter_data/Chinese_data.txt','w',encoding='utf-8') as Write:
                Write.write(str(Chinese_data))
        else:
            with open('../01Create_parameter_data/English_data.txt','r',encoding='utf-8') as Read:
                English_data = Read.read()
            with open('../01Create_parameter_data/Chinese_data.txt','r',encoding='utf-8') as Read:
                Chinese_data = Read.read()
        model_English = Word2Vec(sentences = English_data, vector_size = 100,min_count = 1,workers = 6)
        model_Chinese = Word2Vec(sentences = Chinese_data, vector_size = 100,min_count = 1,workers = 6)
        Chinese_index_to_key = model_Chinese.wv.index_to_key   # 数据类型为list
        Chinese_key_to_index = model_Chinese.wv.key_to_index   # 数据类型为dict
        English_index_to_key = model_English.wv.index_to_key   # 数据类型为list
        English_key_to_index = model_English.wv.key_to_index   # 数据类型为dict
        with open('../01Create_parameter_data/中文（原文-索引）.pkl','wb') as f_chinese:
            pickle.dump(Chinese_index_to_key,f_chinese)
            pickle.dump(Chinese_key_to_index,f_chinese)
        with open('../01Create_parameter_data/英文（原文-索引）.pkl','wb') as f_english:
            pickle.dump(English_index_to_key,f_english)
            pickle.dump(English_key_to_index,f_english)
    else:
        with open('../01Create_parameter_data/中文（原文-索引）.pkl','rb') as f_chinese:
            Chinese_index_to_key = pickle.load(f_chinese)
            Chinese_key_to_index = pickle.load(f_chinese)
        with open('../01Create_parameter_data/英文（原文-索引）.pkl','rb') as f_english:
            English_index_to_key = pickle.load(f_english)
            English_key_to_index = pickle.load(f_english)
    return Chinese_key_to_index,Chinese_index_to_key,English_key_to_index,English_index_to_key

def get_data(file_path = '../00data/translate.csv',num = None):
    data = pd.read_csv(file_path)
    English_data = data.iloc[:, [0]]
    Chinese_data = data.iloc[:, [1]]
    if num == None:
        return English_data,Chinese_data
    else:
        English_data = English_data.iloc[0:num,:]
        Chinese_data = Chinese_data.iloc[0:num,:]
    English_data = np.array(English_data).reshape(-1).tolist()
    Chinese_data = np.array(Chinese_data).reshape(-1).tolist()
    return English_data,Chinese_data

def translate(sentence):
    '预测过程 == 翻译过程'
    global English_key_to_index,Chinese_key_to_index,Chinese_index_to_key,Net     #申明 全局变量
    English_index = torch.tensor([[English_key_to_index[i] for i in sentence]])
    encoder_hidden = Net.encoder(English_index)
    decoder_input  = torch.tensor([[Chinese_key_to_index["<BOS>"]]])   # 编码层的初始隐藏状态的输入
    decoder_hidden = encoder_hidden
    result = []
    while True:
        decoder_output,decoder_hidden = Net.decoder(decoder_input,decoder_hidden)
        pre = Net.linear(decoder_output)
        word_index = torch.argmax(pre , dim = -1)      # 关键步骤： 意思是取最后的线性输出层的最大值的索引
        word = Chinese_index_to_key[word_index]
        if word == '<EOS>' or len(result) > 50:
            break
        else:
            result.append(word)

        decoder_input = torch.tensor([[word_index]])  # 输入字更新
    print('译文：',''.join(result))

class Mydataset(Dataset):
    def __init__(self,English_data,Chinese_data,English_key_to_index,Chinese_key_to_index):
        self.English_data = English_data
        self.Chinese_data = Chinese_data
        self.English_key_to_index = English_key_to_index
        self.Chinese_key_to_index = Chinese_key_to_index

    def __getitem__(self, index):
        English = self.English_data[index]
        Chinese = self.Chinese_data[index]
        English_index = [self.English_key_to_index[i] for i in English]   # 获取英文对应的索引 （index）
        Chinese_index = [self.Chinese_key_to_index[i] for i in Chinese]   # 获取中文对应的索引 （index）
        return English_index , Chinese_index

    def batch_data_process(self,batch_datas):
        # 批量数据处理：进行数据填充,按照 batch_datas 的大小进行填充
        '对__getitem__中传递过来的数据进行处理'
        English_data = []
        Chinese_data = []
        English_len = []
        Chinese_len = []

        for En , Ch in batch_datas:
            English_data.append(En)
            Chinese_data.append(Ch)
            English_len.append(len(En))
            Chinese_len.append(len(Ch))

        Max_English_len = max(English_len)
        Max_Chinese_len = max(Chinese_len)
        "<PAD> 表示填充字符 ， <BOS> 表示开始字符 ， <EOS> 表示结束字符"
        English_data = [ i + [self.English_key_to_index["<PAD>"]] * (Max_English_len - len(i)) for i in English_data]
        Chinese_data = [ [self.Chinese_key_to_index["<BOS>"]] + i + [self.Chinese_key_to_index["<EOS>"]] + [self.Chinese_key_to_index["<PAD>"]] * (Max_Chinese_len - len(i))  for i in Chinese_data]
        English_data = torch.LongTensor(English_data)  # 将 English_data转换为long类型的tensor，为后面的 nn.Enbeding 层作准备
        Chinese_data = torch.LongTensor(Chinese_data)  # 将 Chinese_data转换为long类型的tensor，为后面的 nn.Enbeding 层作准备
        return  English_data, Chinese_data

    def __len__(self):
        len_English_data = len(self.English_data)
        len_Chinese_data = len(self.Chinese_data)
        assert len_English_data == len_Chinese_data
        return len_English_data

class Encoder(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,English_lexicon_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(English_lexicon_len , encoder_embedding_num)  # 编码层，进行词向量编码
        self.lstm = nn.LSTM(input_size = encoder_embedding_num , hidden_size = encoder_hidden_num , batch_first = True)

    def forward(self,English_index):
        English_enbedding = self.embedding(English_index)
        encoder_output , encoder_hidden = self.lstm(English_enbedding)  # 此处的 encoder_output 是不需要的
        return encoder_hidden

class Decoder(nn.Module):
    def __init__(self,decoder_embedding_num,decoder_hidden_num,Chinese_lexicon_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(Chinese_lexicon_len , decoder_embedding_num)
        self.lstm = nn.LSTM(input_size = decoder_embedding_num , hidden_size = decoder_hidden_num , batch_first = True)

    def forward(self,decoder_input,hidden):
        Chinese_embedding = self.embedding(decoder_input)
        decoder_output , decoder_hidden = self.lstm(Chinese_embedding,hidden)
        return decoder_output,decoder_hidden

class Seq2Seq(nn.Module):
    def __init__(self,encoder_embedding_num,encoder_hidden_num,English_lexicon_len,decoder_embedding_num,decoder_hidden_num,Chinese_lexicon_len):
        super(Seq2Seq, self).__init__()
        self.encoder = Encoder(encoder_embedding_num = encoder_embedding_num,
                               encoder_hidden_num = encoder_hidden_num,
                               English_lexicon_len = English_lexicon_len)
        self.decoder = Decoder(decoder_embedding_num = decoder_embedding_num,
                               decoder_hidden_num = decoder_hidden_num,
                               Chinese_lexicon_len = Chinese_lexicon_len)
        self.linear  = nn.Linear(decoder_hidden_num,Chinese_lexicon_len)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self,English_index,Chinese_index):
        encoder_hidden = self.encoder(English_index)     # 编码层的隐藏层输出 = 解码层的隐藏层初始输入
        decoder_input  = Chinese_index[:,:-1]            # 解码层的输入
        label = Chinese_index[:,1:]                      # 标签
        decoder_output , decoder_hidden = self.decoder(decoder_input,encoder_hidden)  # 此处的 decoder_hidden 不使用
        pre = self.linear(decoder_output)
        loss =self.cross_loss(pre.reshape(-1,pre.shape[-1]),label.reshape(-1))
        return loss

if __name__ == '__main__':
    '参数定义'
    epochs = 4
    batch_size = 30
    lr = 0.001
    encoder_embedding_num = 50
    encoder_hidden_num = 100
    decoder_embedding_num = 57
    decoder_hidden_num = 100

    Chinese_key_to_index , Chinese_index_to_key , English_key_to_index , English_index_to_key = Create_parameter_data()
    Chinese_lexicon_len = len(Chinese_key_to_index)  # 中文词库的大小
    English_lexicon_len = len(English_key_to_index)  # 英文词库的大小

    Chinese_key_to_index.update({"<PAD>":Chinese_lexicon_len,"<BOS>":Chinese_lexicon_len + 1,"<EOS>":Chinese_lexicon_len + 2})
    English_key_to_index.update({"<PAD>":English_lexicon_len})
    Chinese_index_to_key = Chinese_index_to_key + ["<PAD>","<BOS>","<EOS>"]
    English_index_to_key = English_index_to_key + ["<PAD>"]

    Chinese_lexicon_len = len(Chinese_key_to_index)  # 中文词库的大小,对前面的结果进行更新，因为添加了元素
    English_lexicon_len = len(English_key_to_index)  # 英文词库的大小,对前面的结果进行更新，因为添加了元素

    English_data , Chinese_data = get_data(num = 1000)
    dataset = Mydataset(English_data = English_data ,
                        Chinese_data = Chinese_data ,
                        English_key_to_index = English_key_to_index ,
                        Chinese_key_to_index = Chinese_key_to_index)
    dataloader = DataLoader(dataset = dataset,batch_size = batch_size,shuffle = False,collate_fn = dataset.batch_data_process)

    Net = Seq2Seq(encoder_embedding_num = encoder_embedding_num,
                  encoder_hidden_num = encoder_hidden_num,
                  English_lexicon_len = English_lexicon_len,
                  decoder_embedding_num = decoder_embedding_num,
                  decoder_hidden_num = decoder_hidden_num,
                  Chinese_lexicon_len = Chinese_lexicon_len)

    optim = torch.optim.Adam(Net.parameters(),lr = lr)
    loss_list = []
    if os.path.exists('../03model/基于Seq2Seq模型的机器翻译') == False:
        start = time.time()
        '训练模型的迭代过程'
        for epoch in range(epochs):
            for i,(English_index,Chinese_index) in enumerate(dataloader):
                loss = Net(English_index,Chinese_index)
                loss.backward()
                optim.step()
                optim.zero_grad()
            print('epoch:'+str(epoch),'loss:'+str(loss))
            loss_list.append(loss.detach().numpy())
        end = time.time()
        total_time = end - start   # 模型训练花费的时间
        print('模型训练完成，一共花费'+str(total_time)+'秒')

        '可视化：loss函数'
        x = np.arange(len(loss_list))
        plt.plot(x,loss_list)
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.title('function:loss')
        plt.savefig('../04picture/loss.svg')
        plt.show()

        '保存模型'
        torch.save(Net.state_dict(),'../03model/基于Seq2Seq模型的机器翻译')

    else:

        '加载模型'
        Net.load_state_dict(torch.load('../03model/基于Seq2Seq模型的机器翻译'))
        Net.eval()

        '翻译过程（预测过程）'
        while True:
            s = input('请输入英文：')
            translate(s)


