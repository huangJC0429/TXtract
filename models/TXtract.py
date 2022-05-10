"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5
# @Author  : Huang Jincheng
# @FileName: TXtract.py
# @describe: TXtract 模型
===========================
"""
from .basic_module import BasicModule
from pytorch_transformers import BertModel
import torch
import torch.nn.functional as F
import math
from .CRFlayer import CRF
import torch.nn as nn



# class LayerNorm(nn.Module):
#     def __init__(self, features, eps=1e-6):
#         super(LayerNorm, self).__init__()
#         self.a_2 = nn.Parameter(torch.ones(features))
#         self.b_2 = nn.Parameter(torch.zeros(features))
#         self.eps = eps
#
#     def forward(self, x):
#         # 就是在统计每个样本所有维度的值，求均值和方差，所以就是在hidden dim上操作
#         # 相当于变成[bsz*max_len, hidden_dim], 然后再转回来, 保持是三维
#         mean = x.mean(-1, keepdim=True)  # mean: [bsz, max_len, 1]
#         std = x.std(-1, keepdim=True)  # std: [bsz, max_len, 1]
#         # 注意这里也在最后一个维度发生了广播
#         return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class TXtract(BasicModule):
    def __init__(self, opt, *args, **kwargs):
        super(TXtract, self).__init__(*args, **kwargs)

        self.model_name = 'TXtract'
        self.opt = opt
        self.poincare_dim = opt.poincare_size
        self.embedding_dim = opt.embedding_dim
        self.hidden_dim = opt.hidden_dim
        self.cond_att_dim = opt.cond_att_dim
        self.tagset_size = opt.tagset_size
        self.num_product = opt.num_product
        self.att_dim=6
        self.batchsize=opt.batch_size

        # for poincare
        self.poincare_vectors = opt.poincare_vectors
        self.W1 = nn.Parameter(torch.empty(size=(self.hidden_dim, self.cond_att_dim)))
        nn.init.xavier_uniform_(self.W1.data, gain=1.414)
        self.W2 = nn.Parameter(torch.empty(size=(self.hidden_dim, self.cond_att_dim)))
        nn.init.xavier_uniform_(self.W2.data, gain=1.414)
        self.W3 = nn.Parameter(torch.empty(size=(self.poincare_dim, self.cond_att_dim)))
        nn.init.xavier_uniform_(self.W3.data, gain=1.414)
        self.W_a = nn.Parameter(torch.empty(size=(self.cond_att_dim, 1)))
        nn.init.xavier_uniform_(self.W_a.data, gain=1.414)

        self.b_g = nn.Parameter(torch.empty(size=(1, self.cond_att_dim)))
        nn.init.xavier_uniform_(self.b_g.data, gain=1.414)
        self.b_a = nn.Parameter(torch.empty(size=(1, 1)))
        nn.init.xavier_uniform_(self.b_a.data, gain=1.414)

        # Product Category Prediction for att
        self.linear_att = nn.Linear(self.hidden_dim, self.cond_att_dim)
        self.u_c = nn.Parameter(torch.empty(size=(self.cond_att_dim, 1)))
        nn.init.xavier_uniform_(self.u_c.data, gain=1.414)

        self.category_classifier = nn.Linear(self.hidden_dim, self.num_product)

        self.bert = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.word_embeds = torch.nn.Embedding(30000, self.opt.embedding_dim)

        # self.W_Q = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.W_K = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        # self.W_V = torch.nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        self.dropout = torch.nn.Dropout(opt.dropout)

        self.LN = torch.nn.LayerNorm([40,1024],elementwise_affine=True)

        #CRF
        self.lstm = torch.nn.LSTM(self.embedding_dim,
                                  self.hidden_dim // 2,
                                  num_layers=1,
                                  bidirectional=True,
                                  batch_first=True)

        # self.hidden2tag = torch.nn.Linear(self.hidden_dim * 2,self.tagset_size)
        self.hidden2tag = torch.nn.Linear(self.hidden_dim, self.tagset_size) # 线性激活函数
        self.crf = CRF(self.tagset_size, batch_first=True)

        # 用来计算attention
    def Self_Attention(self, Q, K, V):
        d_k = K.size(-1)
        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(d_k)
        alpha_n = F.softmax(scores, dim=-1)
        context = torch.matmul(alpha_n, V)

        # 这里都是组合之后的矩阵之间的计算，所以.sum之后，得到的output维度就是[batch_size,hidden_dim]，并且每一行向量就表示一句话，所以总共会有batch_size行
        # output = context.sum(1)

        return context, alpha_n

    def CondSelfAtt(self, h, e_c):
        # TXtract中的条件自注意力, h [batch_size, seq_len, hidden_dim],下面的变量名都是遵循论文中的名字的可以对照观看
        re1 = h.repeat(1, h.shape[1], 1)  # 重复为 1,2,1,2 ,dim = [batch_size, seq_len*seq_len, hidden_dim]
        re2 = h.repeat_interleave(h.shape[1], dim=1) # 重复为 1,1,2,2 ,dim = [batch_size, seq_len*seq_len, hidden_dim]
        poincare = torch.repeat_interleave(e_c.unsqueeze(dim=1), repeats=h.shape[1]**2, dim=1)  # 将二维的复制编程3维

        g = torch.tanh(torch.matmul(re1, self.W1) + torch.matmul(re2, self.W2) + torch.matmul(poincare, self.W3) + self.b_g)
        # print(g.shape)  # torch.Size([32, 1600, 50])
        '''
        下面这一步是否可以不用sigmoid改为使用softmax归一化或者batchnorm归一化，因为用sigmoid会导致被放大或者缩小。可尝试
        '''
        A = torch.sigmoid(torch.matmul(g, self.W_a).squeeze() + self.b_a)  # 32, 1600 batch, s*s 32个batch的注意力分数，分别是40*40
        A = A.reshape(-1, 40, 40) # 这个40是固定了的，一共有40个token
        h_wave = torch.matmul(A, h)  # 按照注意力矩阵加权求和

        return h_wave

    def CategoryEnc(self, poincare_index):
        # 实现TXtract中的CategoryEnc，这里的实现思路就是去查找对应vector的庞加莱嵌入向量
        return self.poincare_vectors[poincare_index]  # batch, c 这里c是庞加莱嵌入的维度opt.poincare_size


    def attention_layer(self, h):
        # Product Category Prediction端的attention
        # torch.matmul(h, self.W_c)
        # print(h.shape)
        beta = h.contiguous().view(-1, self.hidden_dim).contiguous()  # [batch_size*seq_len, hidden_dim]
        beta = torch.tanh(self.linear_att(beta))  # [batch_size*seq_len, cond_att_dim]
        beta = beta.view(-1, 40, self.cond_att_dim)
        beta = torch.matmul(beta, self.u_c)  # [batch_size, seq_len, 1]
        beta = torch.softmax(beta, dim=1) # [batch_size, seq_len, 1]
        beta = torch.transpose(beta, 1, 2)  # [batch_size, 1, seq_len]
        h_out = torch.matmul(beta, h) # [batch_size, 1, hidden_dim]
        return h_out

    def CategoryCLF(self, h_out):
        # h_out.shape = [batchsize, 1, hidden_dim]
        h_out = torch.squeeze(h_out)  # [batchsize, hidden_dim]
        y_hat = self.category_classifier(h_out)  # [batchsize, num_category]
        # c_hat = torch.sigmoid(y_hat)
        return y_hat  # 输出sigmoid多任务二分类概率


    def forward(self, inputs):  # 模型预测
        context, att, target, poincare_index = inputs[0], inputs[1], inputs[2], inputs[3]
        # 模型预测是先预测标签种类
        context, _ = self.bert(context)
        context_output, (final_hidden_state, final_cell_state) = self.lstm(context)  # con.. ([32, 40, 1024])
        '''
        Taxonomy-Aware Product Category Prediction
        '''
        h_out = self.attention_layer(context_output)
        c_hat = self.CategoryCLF(h_out)  # [batchsize, num_category]
        # print(c_hat)
        c_hat = torch.argmax(c_hat, dim=1)  # [bathsize, 1]

        e_c = self.CategoryEnc(c_hat)
        '''
              Taxonomy-Aware Attribute Value Extraction, let predicted e_c as real label to mark 'B,I,O'
              '''
        h_wave = self.CondSelfAtt(context_output, e_c)
        outputs = self.dropout(h_wave)  # attn_output: [batchsize,seqlen,hidden_dim]
        # print(outputs.shape)  # 32, 40, 200
        outputs = self.hidden2tag(outputs)
        outputs = self.crf.decode(outputs)  # 这里就是预测值

        return outputs, c_hat

    def log_likelihood(self, inputs):  # forward函数 训练用
        context, att, target, poincare_index = inputs[0], inputs[1], inputs[2], inputs[3]

        context, _ = self.bert(context)
        # print("contex after bert:", context, "shape", context.shape)  # size:(32,40,768)
        context_output, (final_hidden_state, final_cell_state) = self.lstm(context)

        '''
        Taxonomy-Aware Attribute Value Extraction
        '''
        e_c = self.CategoryEnc(poincare_index)
        h_wave = self.CondSelfAtt(context_output, e_c)
        outputs = self.dropout(h_wave)  # attn_output: [batchsize,seqlen,hidden_dim]
        # print(outputs.shape)  # 32, 40, 200
        outputs = self.hidden2tag(outputs)
        loss_crf = -self.crf(outputs, target)  # CRF返回值为对数似然，所以当你作为损失函数时，需要在这个值前添加负号,上面作为预测使用decode即可

        '''
        Taxonomy-Aware Product Category Prediction
        '''
        h_out = self.attention_layer(context_output)
        c_hat = self.CategoryCLF(h_out)
        return loss_crf, c_hat
