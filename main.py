import os
import sys
from time import strftime, localtime
from collections import Counter
from config_TXtract import opt
from pytorch_transformers import BertTokenizer,AutoTokenizer
import random
import numpy as np
import torch
import models
from utils.TXtract_dataset_small_data import get_dataloader,get_test_dataloader,get_dataloader_15w_1w_final,get_1w_dataloader
# from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from seqeval.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm
import datetime
import torch.utils.data as Data	# 用于创建 DataLoader
import torch.nn as nn
import pytorchtools
import torch.nn.functional as F
from pytorchtools import EarlyStopping

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')

multi_task_y = torch.tensor(np.load('./Data/品类树三极目录嵌入/庞加莱圆盘嵌入/y.npy')).float().to(opt.device)
multi_task_wi = torch.tensor(np.load('./Data/品类树三极目录嵌入/庞加莱圆盘嵌入/wi.npy')).float().to(opt.device)

class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight, power, reduction='mean'):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = torch.pow(pos_weight, power)  # 这里为论文中的超参数权重w[n, num_category]
        self.reduction = reduction

    def forward(self, logits, target, poincare_index):
        # logits: [N, *], target: [N, *]
        # print(logits)
        # print(target)
        logits = torch.sigmoid(logits)
        pos_weight = self.pos_weight[poincare_index]
        loss = - target * torch.log(logits) - \
               (1 - target) * torch.log(1 - logits)
        loss = torch.multiply(pos_weight, loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss

def compute_acc(y_hat, y_true):
    # 维度都是[batch, 1]
    print(y_hat)  # (4,)
    print(y_true)  # (4,)
    correct = y_hat.eq(y_true).double()
    correct = correct.sum()
    return correct / len(y_true)

def get_attributes(path):
    values = {}
    for root, dirs, files in os.walk(path):
        # root="/Data/碳酸水"
        # files=os.listdir(root)
        for file in files:
            with open(os.path.join(root, file), encoding='utf-8') as f:
                for line in f.readlines():
                    value = line.split('"')[1]
                    attibute = line.split('"')[13]
                    if attibute== '' or value=='':
                        continue
                    if value in values.keys():
                        values[value] = values[value] + 1
                    else:
                        count = 1
                        values[value] = count
    return values.keys()

def test(model,test_dataloader,id2tags):
    # 4.2 验证模型
    print("开始验证")
    preds, labels,title = [], [],[]
    true_count,false_count=0,0
    true_sample,false_sample=[],[]
    count=0
    # 注释掉的意思就是将结果保存到f文件中
    # sys.stdout = f
    # sys.stderr = f
    model.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(test_dataloader):
            x = batch['x'].to(opt.device)
            y = batch['y'].to(opt.device)
            att = batch['att'].to(opt.device)
            inputs = [x, att, y]
            predict = model(inputs)
            # 1. 统计非0的，也就是真实标签的长度
            leng = []
            for i in y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)
            # 2. 获得预测标签集合
            for index, i in enumerate(predict):
                m=[id2tags[k] if k > 0 else 'O' for k in i]
                preds.append(m)
                p=[id2tags[k] if k > 0 else 'O' for k in y.cpu().tolist()[index]]
                if m==p:
                    true_count += 1
                    true_sample.append([m,count])
                elif m != p:
                    false_count += 1
                    false_sample.append([m,count])
                count+=1
            # 3. 获得真实标签集合
            for index, i in enumerate(y.tolist()):
                m=[id2tags[k] if k > 0 else 'O' for k in i]
                labels.append(m)
            # 4.解码title
            for line in x.cpu():
                m=tokenizer.convert_ids_to_tokens(line.cpu().tolist())
                title.append(m)
        # 4. calculate F1,precision,recall on entity,返回评价报告
        report = classification_report(preds, labels)
        report = report.strip().split()
        precision = float(report[5])
        recall = float(report[6])
        f1 = float(report[7])

def valid(model,valid_dataloader,id2tags,epoch,best_f1,best_p,best_r):
    # 4.2 验证模型
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for bidx, batch in enumerate(valid_dataloader):
            x = batch['x'].to(opt.device)
            y = batch['y'].to(opt.device)
            att = batch['att'].to(opt.device)
            poincare_index = batch['poincare_index'].to(opt.device)  # 获取商品的种类，可以通过这个数查询庞加莱嵌入

            c_label = torch.argmin(multi_task_wi[poincare_index],dim=1) # 获取准确标签

            inputs = [x, att, y, poincare_index]
            predict, c_hat = model(inputs)
            # print("predict c_hat:", c_hat)
            # 1. 统计非0的，也就是真实标签的长度
            leng = []
            for i in y.cpu():
                tmp = []
                for j in i:
                    if j.item() > 0:
                        tmp.append(j.item())
                leng.append(tmp)
            # 2. 获得预测标签集合
            for index, i in enumerate(predict):
                preds.append([id2tags[k] if k > 0 else 'O' for k in i[:len(leng[index])]])
            # 3. 获得真实标签集合
            for index, i in enumerate(y.tolist()):
                labels.append([id2tags[k] if k > 0 else 'O' for k in i[:len(leng[index])]])
        # 4. calculate F1,precision,recall on entity,返回评价报告
        report = classification_report(preds, labels)
        report = report.strip().split()
        precision = float(report[5])
        recall = float(report[6])
        f1 = float(report[7])

        #2. 计算 classification
        acc = compute_acc(c_hat, c_label)

        model.train()
        print("此step时间为：{} [valid] epoch {} f1 {} precision {} recall {}, 品类分类准确率为 {}".format(datetime.datetime.now(),epoch, f1, precision,recall, acc))

        is_updatemodel = False
        # 这里原来的模型选择代码有点问题
        flag = False
        # 5.如果P,F,R比前几轮的更好，则更新并保存新模型
        if f1 > best_f1 or precision > best_p or recall > best_r:
            if f1 > best_f1:
                if precision >= best_p and recall >= best_r:
                    flag = True
            elif precision > best_p:
                if f1 >= best_f1 and recall >= best_r:
                    flag = True
            elif recall > best_r:
                if precision >= best_p and f1 >= best_f1:
                    flag = True
            if flag:
                best_f1 = f1
                best_p = precision
                best_r = recall
                is_updatemodel = True
                print('best_f1为：{},best_P为：{},best_R为：{}'.format(best_f1, best_p, best_r))
    return best_f1, best_p, best_r, f1, precision, recall, is_updatemodel
def train(**kwargs):
    test_flag = opt.test  # 是否打印具体的预测结果，如果需要利用已经训练好的模型测试数据并打印数据预测正确和错误的情况，test_flag可设置为True
    tags2id = {'B': 1, 'I': 2,'O': 3,'PAD':0}    # 定义标签字典
    id2tags = {v: k for k, v in tags2id.items()}

    poincare_vectors = np.load('./Data/品类树三级目录嵌入/庞加莱圆盘嵌入/product_vectors.npy')

    opt._parse(kwargs)

    if opt.seed is not None:                     # 设置随机数种子
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    poincare_vectors = torch.tensor(poincare_vectors).float().to(opt.device)
    opt.poincare_vectors = poincare_vectors
    # step1: configure model
    model = getattr(models, opt.model)(opt)

    # step2: load data
    if test_flag:
        train_dataloader, test_dataloader, length_testdata = get_1w_dataloader(opt)
    else:
        train_dataloader, valid_dataloader,length_data = get_dataloader_15w_1w_final(opt)

    # step3: get optimizer
    lr = opt.lr
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # opt.load_model_path='./checkpoints/Bert_LSTM_selfAttention_数据142805条_epoch1'
    # opt.load_model_path = './checkpoints/' + opt.model + '_数据' + str(151339) +'条_epoch' + str(7)
    # opt.load_model_path = './checkpoints/' + opt.model + '_数据' + str(length_data)    # 保存最好训练模型的路径

    # 测试的时候这里要改成9413
    # opt.load_model_path = './checkpoints/' + opt.model + '_数据' + str(9413)  # 保存最好训练模型的路径
    # opt.load_model_path = './checkpoints/' + opt.model + 'LN+Residual_数据' + str(9413)  # 保存最好训练模型的路径
    opt.load_model_path = './checkpoints/' + opt.model + 'TXtract_数据' + str(9413)  # 保存最好训练模型的路径
    # test_flag=True时执行下面
    # 打印测试样例正确与错误的结果。如果test_flag=True,则加载已保存的最好模型
    if test_flag:
        # 加载保存的模型直接进行测试机验证，不进行此模块以后的步骤
        checkpoint = torch.load(opt.load_model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        model.to(opt.device)
        test(model, test_dataloader, id2tags)
        return
    loss_product_tree = WeightedBCELoss(opt.w, multi_task_wi)  # 这个内部没有sigmoid所以要先计算sigmoid
    # step4 load model
    # 4.1 如果有保存的模型，则加载模型，并在其基础上继续训练，否则从头开始训练。
    print(opt.load_model_path)
    print(os.path.exists(opt.load_model_path))
    if os.path.exists(opt.load_model_path):
        checkpoint = torch.load(opt.load_model_path,map_location=torch.device(opt.device))
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')
    model.to(opt.device)

    # step5 train
    print("开始训练的时间为：", datetime.datetime.now())
    best_f1,best_p,best_r = 0.0,0.0,0.0
    patience = 4  # 当验证集损失在连续4次训练周期中都没有得到降低时，停止模型训练，以防止模型过拟合
    early_stopping = EarlyStopping(patience, verbose=True)
    for epoch in range(start_epoch + 1, opt.max_epoch):
        model.train()
        for ii, batch in tqdm(enumerate(train_dataloader)):
            optimizer.zero_grad()              # 每一轮开始，梯度清零，防止上一轮数据影响
            x = batch['x'].to(opt.device)      # 获得title编码后的数据
            y = batch['y'].to(opt.device)      # 获得title的标签编码后的数据
            att = batch['att'].to(opt.device)  # 获得品牌名的编码后的数据
            poincare_index = batch['poincare_index'].to(opt.device)  # 获取商品的种类，可以通过这个数查询庞加莱嵌入
            m_y = multi_task_y[poincare_index]  # 32, 576

            inputs = [x, att, y, poincare_index]

            # 5.1 训练模型
            loss_crf, c_hat = model.log_likelihood(inputs)# .to(opt.device)  # 前向传播，预测数据，然后根据预测和标签比较，计算损失
            loss_predict = loss_product_tree(c_hat, m_y, poincare_index)
            # print(loss_predict)
            # exit()
            loss = opt.gamma*loss_crf + (1-opt.gamma)*loss_predict
            loss.backward()    # 反向传播，更新梯度
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=3) # 梯度裁剪
            optimizer.step()   # 梯度下降后，优化器更新网络的参数
            if ii % opt.print_freq == 0:  # 每print_freq个step打印一次模型，并打印f1, precision, recall
                print('epoch:%04d,step:%d,------------loss:%f' % (epoch,ii, loss.item()))
        # 5.2 验证模型
        best_f1, best_p, best_r, valid_f1, valid_precision, valid_recall,is_update = valid(model, valid_dataloader, id2tags, epoch, best_f1,best_p,best_r)
        # 5.3 更新并保存模型
        if is_update:
            # opt.load_model_path = './checkpoints/' + opt.model + '_数据' + str(length_data)
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),'optimizer_state_dict': optimizer.state_dict(), }, opt.load_model_path)

        if early_stopping.early_stop:  # 早停
            print("Early stopping")
            # 提前结束模型训练
            break
    print('==========================')


if __name__ == '__main__':
    train()
