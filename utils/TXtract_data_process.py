"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/5/5
# @Author  : Huang Jincheng
# @FileName: construct_tree.py
# @describe: TXtract label数据处理
===========================
"""
import numpy as np
import pandas as pd
import os

# # 构建TXtract的label，测试代码
# three_index = []
# path = '../Data/原数据'
# tree_root = '总数据'
#
# # 遍历一级目录的文件夹
# for root, dirs, files in os.walk(path):
#     print("root:", root)
#     print("dirs:", dirs)
#     print("files:", files)
#     for i in range(len(dirs)):
#         three_index.append(dirs[i])
#     # 遍历二级目录的文件夹
#     for d in dirs:
#         path1 = os.path.join(root, d)
#         # print(d)
#         for root1, dirs1, files1 in os.walk(path1):
#             for i in range(len(dirs1)):
#                 three_index.append(dirs1[i])
#             # print(dirs1)
#             # print(three_struct)
#             # exit()
# #            # print("dirs:", dirs1)
#            # 遍历三级目录的文件夹
#             for d1 in dirs1:
#                 path2 = os.path.join(root1, d1)
#                 for root2, dirs2, files2 in os.walk(path2):
#                     for i in range(len(dirs2)):
#                         three_index.append(dirs2[i])
#                     break
#             break
#     break
# three_index.insert(0, '总数据')
# print(three_index)
# print(len(three_index))  # 一共576个连边关系
# # np.save('../Data/品类树三级目录嵌入/树结构.npy', three_struct)
#
# B = np.load('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/product_names.npy').tolist()
# print(B)
# for i in range(len(B)):
#     if B[i] != three_index[i]:
#         print(B[i])
#         print(three_index[i])
#         print("error")
# print('相同')




# 构建TXtract的label
multi_sigmoid_label = []
hop_weight = []

aa = np.zeros((576))
aa[0] = 1
multi_sigmoid_label.append(aa)
hop_weight.append(aa)
path = '../Data/原数据'
tree_root = '总数据'

product_names = np.load('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/product_names.npy').tolist()
# 根据 sparse_index 建立稠密矩阵muti_label和主路距离
def sparse_to_dense(sparse_index):
    res = np.zeros(576)
    wi = np.ones(576)*3  # 0, 1, 2, 3
    for i in range(len(sparse_index)):
        res[sparse_index[i]] = 1
        wi[sparse_index[i]] = len(sparse_index)-1-i  #主路距离
    return res, wi
# 遍历一级目录的文件夹
for root, dirs, files in os.walk(path):
    print("root:", root)
    print("dirs:", dirs)
    print("files:", files)
    for i in range(len(dirs)):
        index = product_names.index(dirs[i])
        sparse_index = [index]

        label, wi = sparse_to_dense(sparse_index)
        multi_sigmoid_label.append(label)
        hop_weight.append(wi)
        # three_index.append(dirs[i])
    # 遍历二级目录的文件夹
    for d in dirs:
        path1 = os.path.join(root, d)
        # print(d)
        # exit()
        index1 = product_names.index(d)
        for root1, dirs1, files1 in os.walk(path1):
            for i in range(len(dirs1)):
                index = product_names.index(dirs1[i])
                sparse_index = [index1, index]

                label, wi = sparse_to_dense(sparse_index)
                multi_sigmoid_label.append(label)
                hop_weight.append(wi)
            # print(dirs1)
            # print(three_struct)
            # exit()
#            # print("dirs:", dirs1)
           # 遍历三级目录的文件夹
            for d1 in dirs1:
                index2 = product_names.index(d1)
                path2 = os.path.join(root1, d1)
                for root2, dirs2, files2 in os.walk(path2):
                    for i in range(len(dirs2)):
                        index = product_names.index(dirs2[i])
                        sparse_index = [index1, index2, index]

                        label, wi = sparse_to_dense(sparse_index)
                        multi_sigmoid_label.append(label)
                        hop_weight.append(wi)
                    break
            break
    break
# 最后得到的 multi_sigmoid_label就是，类别到根节点路径上的品类都是1，其他都是0，hop_weight为1的标签中，离目标节点的距离
print(multi_sigmoid_label[0].shape)
y = np.array(multi_sigmoid_label)
wi = np.array(hop_weight)
print(y.shape)
print(wi.shape)
# for i in range(len(wi)):
#     print(wi[i])
# exit()
np.save('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/y.npy', y)
np.save('../Data/品类树三级目录嵌入/庞加莱圆盘嵌入/wi.npy', wi)