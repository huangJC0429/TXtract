import numpy as np
import pandas as pd
from gensim.models.poincare import PoincareModel
import sys,os
sys.path.append(os.path.dirname(__file__) + os.sep + '../')
from config_TXtract import  opt


tree_struct = np.load('../Data/品类树三级目录嵌入/树结构.npy')
# # 先可视化一下品类树
# src = []
# dst = []
# for i in range(len(tree_struct)):
#     src.append(tree_struct[i][0])
#     dst.append(tree_struct[i][1])
# tree = pd.DataFrame()
# tree['Source'] = src
# tree['Target'] = dst
# tree.to_csv('../Data/品类树三级目录嵌入/树结构用于Gephi.csv')



# relations = [('kangaroo', 'marsupial'), ('kangaroo', 'mammal'), ('gib', 'cat')]
relations = tree_struct
model = PoincareModel(relations, negative=2, size=opt.poincare_size)
model.train(epochs=50)
# 输出嵌入的向量
# print(model.kv['包子'])
# print(model.kv['手抓饼'])
# print(model.kv['海外直采啤酒'])
# print(len(model.kv))
# exit()
product_names = []  # 每个商品名称
product_vectors = []  # 对应的嵌入向量
for i in model.kv.index_to_key:
    product_names.append(i)
    product_vectors.append(model.kv[i])
print(len(product_names))
print(len(product_vectors[0]))
np.save('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/product_names.npy', product_names)
np.save('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/product_vectors.npy', product_vectors)