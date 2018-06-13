# import tensorflow as tf
#
# import tensorflow as tf
# import numpy as np
# import pandas as pd
#
# def addLayer(inputData, inSize, outSize, activity_function=None):
#     Weights = tf.Variable(tf.random_normal([inSize, outSize]))
#     basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
#     weights_plus_b = tf.matmul(inputData, Weights) + basis
#     if activity_function is None:
#         ans = weights_plus_b
#     else:
#         ans = activity_function(weights_plus_b)
#     return ans
#
#
# path = 'data/game_info.csv'
# path2 = 'data/game_info_test.csv'
# data = pd.read_csv(path)
# x_data = data.values[:, 5:10].tolist()
# y_data = data.values[:, 14:16].tolist()
#
# data2 = pd.read_csv(path2)
# x_data2 = data2.values[:, 5:10].tolist()
# y_data2 = data2.values[:, 14:16].tolist()
#
# xs = tf.placeholder(tf.float32, [None, 5])  # 样本数未知，特征数为1，占位符最后要以字典形式在运行中填入
# ys = tf.placeholder(tf.float32, [None, 2])
#
# l1 = addLayer(xs, 5, 6, activity_function=tf.nn.relu)  # relu是激励函数的一种
# l2 = addLayer(l1, 6, 2, activity_function=None)
# loss = tf.reduce_mean(tf.reduce_sum(tf.square((ys - l2)), reduction_indices=[1]))  # 需要向相加索引号，redeuc执行跨纬度操作
#
# train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 选择梯度下降法
#
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
#
# for i in range(10000):
#     sess.run(train, feed_dict={xs: x_data, ys: y_data})
#     if i % 50 == 0:
#         print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing
from collections import Counter
#
# # sess = tf.InteractiveSession()
# path = 'data/game_info.csv'
# data = pd.read_csv(path)#dateframe格式
#
# n_input = 5
# # n_output = 14
# #学习率
# learning_rate = 0.1
# training_step = 200
# batch_size = 100
#
# X = data.values[:, 5:10].tolist()
# Y = data.values[:, 14:16].tolist()
# print(X)
#
# Y2 = []
# for i in Y :
#     if i not in Y2:
#         Y2.append(i)
# print(len(Y2),Y2)


str_l ='{'
l = 0
for i in range(5):
    for j in range(5):
        if i == 4 and j == 4:
            str_l = str_l + "'"+ str(i) + '-' + str(j)+"'" + ':' + str(l) + '}'
        else:

            str_l = str_l + "'"+ str(i) +'-' + str(j) + "'"+':'+ str(l)+','
        l = l+1
# for i in range(5):
#     for j in range(5):
#         if i == j:
#             str_l = str_l + "'"+ str(i) +'-' + str(j) + "'"+':'+ str(l)+','
#             l = l+1
# for i in range(5):
#     for j in range(i):
#         if i == 4 and j == 3:
#             str_l = str_l + "'"+ str(j) + '-' + str(i)+"'" + ':' + str(l) + '}'
#         else:
#             str_l = str_l + "'"+ str(j) +'-' + str(i) + "'"+':'+ str(l)+','
#         l = l+1

print (str_l)
