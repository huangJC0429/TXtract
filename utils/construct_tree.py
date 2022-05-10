"""
============================
# -*- coding: utf-8 -*-
# @Time    : 2022/4/26
# @Author  : Huang Jincheng
# @FileName: construct_tree.py
# @describe: 将品类树转换三级目录提取出来
# @describe: 后续如果效果好，就将品类树转换成数据结构存储，方便后面的硬匹配
===========================
"""
import numpy as np
import pandas as pd
import os

# three_cate = []
# path = '../Data/原数据'
# # 遍历一级目录的文件夹
# for root, dirs, files in os.walk(path):
#     # print("root:", root)
#     # print("dirs:", dirs)
#     # print("files:", files)
#     # 遍历二级目录的文件夹
#     for d in dirs:
#         path1 = os.path.join(root, d)
#         # print(path_temp)
#         for root1, dirs1, files1 in os.walk(path1):
#             # print("dirs:", dirs1)
#             # 遍历三级目录的文件夹
#             for d1 in dirs1:
#                 path2 = os.path.join(root1, d1)
#                 for root2, dirs2, files2 in os.walk(path2):
#                     print("dirs:", dirs2)
#                     three_cate.extend(dirs2)
#                     break
#             break
#     break
# print(len(three_cate))  # 一共504个三级目录
# np.save('../Data/品类树三级目录嵌入/三级目录列表.npy', three_cate)


# 构建树的list[tuple(A,B)],用于庞加莱圆盘嵌入
three_struct = []
path = '../Data/原数据'
tree_root = '总数据'
# 遍历一级目录的文件夹
for root, dirs, files in os.walk(path):
    print("root:", root)
    print("dirs:", dirs)
    print("files:", files)
    for i in range(len(dirs)):
        three_struct.append((tree_root, dirs[i]))
    # 遍历二级目录的文件夹
    for d in dirs:
        path1 = os.path.join(root, d)
        # print(d)
        for root1, dirs1, files1 in os.walk(path1):
            for i in range(len(dirs1)):
                three_struct.append((d, dirs1[i]))
            # print(dirs1)
            # print(three_struct)
            # exit()
#            # print("dirs:", dirs1)
           # 遍历三级目录的文件夹
            for d1 in dirs1:
                path2 = os.path.join(root1, d1)
                for root2, dirs2, files2 in os.walk(path2):
                    for i in range(len(dirs2)):
                        three_struct.append((d1, dirs2[i]))
                    break
            break
    break
print(three_struct)
print(len(three_struct))  # 一共576个连边关系
print(three_struct)
np.save('../Data/品类树三级目录嵌入/树结构.npy', three_struct)