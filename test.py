#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 @Author   : Sword
 @Date     : 2021-09-25 
 @Time     : 17:21
 @Version  : v1.0
 @File     : test.py 
 @Describe :
"""
# import sys
# int_list = []
# for line in sys.stdin:
#
#     int_list = list(map(int, line.split(',')))
#     break
# len_ = len(int_list)
# p = 1
# j = len_ - 1
# is_add = 0
# while j >= 0:
#     # 从个位开始判断
#     if (int_list[j] + p + is_add) >= 10:
#         # 如果加上1之后发生进位的话
#         is_add = 1
#         int_list[j] = 0
#     else:
#         int_list[j] += p + is_add
#         is_add = 0
#         break
#     p = 0
#     j -= 1
# if is_add == 1:
#     res = [is_add] + int_list
#     print(res)
#     sys.exit(0)
# print(int_list)


# s = input()
# ch_list = list(s)
# res = []
# ord_list = list(map(ord, ch_list))
# sum_list = []
# for i in ord_list:
#     int_list = list(map(int, list(str(i))))
#     sum_i = 0
#     for j in int_list:
#         sum_i += j
#     sum_list.append(sum_i)
# for k, number in zip(ch_list, sum_list):
#     print(k * number)

# import sys
# import random
# if __name__ == "__main__":
#     # 读取第一行的n
#     n = int(sys.stdin.readline().strip())
#     ans = []
#     for i in range(n):
#         # 读取每一行
#         line = sys.stdin.readline().strip()
#         # 把每一行的数字分隔后转化成int列表
#         values = list(map(int, line.split()))
#         for v in values:
#             ans.append(v)
#     res = {}
#     for idx, v in enumerate(ans):
#         if (v == 1):
#             res[idx] = 1
#         elif (v == 2):
#             res[idx] = 2
#         else:
#             break
#     res_list = list(res.values())
#     for i in res_list:
#         print(i)

# import sys
# import math
# # 1 1 1 2 2 3 3 1 2
# p1 = []
# p2 = []
# p3 = []
# r = []
# d = 0
# for line in sys.stdin:
#     a = line.split()
#     a = list(map(int, a))
#     p1 = [a[1], a[2]]
#     p2 = [a[3], a[4]]
#     p3 = [a[5], a[6]]
#     r = [a[7], a[8]]
#     d = a[0]
#     break
#
#
# def comp_dist(x, y):
#     d = ((x[0] - y[0])**2 + (x[1] - y[1])**2)**0.5
#     return d
#
# d_p1_r = comp_dist(p1, r)
# d_p2_r = comp_dist(p2, r)
# d_p3_r = comp_dist(p3, r)
# print(d_p1_r)
# print(d_p2_r)
# print(d_p3_r)
# attack = 0
# if d_p1_r <= d:
#     attack += 1
# else:
#     attack += 0
# if d_p2_r <= d:
#     attack += 1
# else:
#     attack += 0
# if d_p3_r <= d:
#     attack += 1
# else:
#     attack += 0
# print(str(attack) + 'x')

# import sys
# # 5 7    35
# # 2 4 -- 4
# # 3 6 -- 6  包含
# # 3 7 -- 21
# # 4 6 -- 12
# # 6 8 -- 48
# for line in sys.stdin:
#     def judge(x):
#         count = 0
#         res = []
#         res.append(1)
#         res.append(x)
#         for i in range(2, x):
#             a = x / i
#             b = str(a).split('.')
#             if (b[1] == '0'):
#                 res.append(i)
#                 count += 1
#         if count > 0:
#             return False, res
#         return True, res
#     def judge_2(max_x, min_x):
#         if min_x == 1:
#             return max_x
#         for i in range(2, min_x + 1):
#             a = max_x * i
#             b = a / min_x
#             c = str(b).split('.')
#             if (c[1] == '0'):
#                 return a
#             else:
#                 continue
#     a = line.split()
#     a = list(map(int, a))
#     x1 = a[0]
#     x2 = a[1]
#     flag_x1, res_x1 = judge(x1)
#     flag_x2, res_x2 = judge(x2)
#     print(res_x1)
#     print(res_x2)
#     print(flag_x1)
#     print(flag_x2)
#     if flag_x1 and flag_x2: # 3 7
#         result = x1 * x2
#         print(result)
#     elif (flag_x1 and not flag_x2):
#         if x1 in res_x2: # 3 6
#             result = x2
#             print(result)
#         else: # 4 15
#             result = x1 * x2
#             print(result)
#     elif (not flag_x1 and flag_x2):
#         if x2 in res_x1:
#             result = x1
#             print(result)
#         else:
#             result = x1 * x2
#             print(result)
#     else:
#         # 15 30
#         if x2 in res_x1:
#             result = x1
#             print(result)
#         elif x1 in res_x2:
#             result = x2
#             print(result)
#         else: # 4 6
#             if (x1 > x2):
#                 result = judge_2(x1, x2)
#                 print(result)
#                 # a = (x1 * 2) / x2
#                 # b = str(a).split('.')
#                 # if (b[1] == '0'):
#                 #     result = x1 * 2
#                 #     print(result)
#                 # else:
#                 #     result = x1 * x2
#                 #     print(result)
#             else:
#                 # a = (x2 * 2) / x1
#                 # b = str(a).split('.')
#                 # if (b[1] == '0'):
#                 #     result = x2 * 2
#                 #     print(result)
#                 # else:
#                 #     result = x1 * x2
#                 #     print(result)
#                 result = judge_2(x1, x2)
#                 print(result)
# import numpy as np
# a = [[1, 2, 3], [3, 2, 1]]
# b = [[1, 1, 1], [1, 1, 1]]
# a = np.array(a)
# b = np.array(b)
# c = np.add(a, b)
# print(c)
import torch as th

# obtain_cat_seq = [1, 2, 3, 4, 5, 1]
# padded_cat_seq = th.zeros(20)
# if len(obtain_cat_seq) < 20:
#     for j in list(range(len(obtain_cat_seq))):
#         padded_cat_seq[j] = obtain_cat_seq[j]
# print(padded_cat_seq)
import math

a = list(range(1, 11))
b = [math.exp((x - max(a)) + math.log2(max(a) + 1)) for x in a]
print(a)
print(max(a))
print(b)
print(sum(b))
c = [(x / sum(b)) for x in b]
print(c)

def get_position_weight(position_list:list):
    b = [math.exp((x - max(position_list)) + math.log2(max(position_list) + 1)) for x in position_list]
    weight_list = [(x / sum(b)) for x in b]
    return weight_list