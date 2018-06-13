import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing

# sess = tf.InteractiveSession()
path = 'data/game_info_win.csv'
data = pd.read_csv(path)#dateframe格式

n_input = 4
# n_output = 14
#学习率
learning_rate = 0.1
training_step = 1000
batch_size = 100

X = data.values[:, 5:9].tolist()
Y = data.values[:, 12:13].tolist()
print(X)
Y_temp = []
min_Y = np.min(Y)
for i in Y:
    Y_temp.append(i[0] - min_Y)
Y = Y_temp
print (Y)
Y_max = np.max(Y)+1
# Y_max = 1
n_output = Y_max

classes = Y_max

n_hidden = 7
n_hidden2 = 7

#神经网络层定义

def inference(x_input):
    with tf.variable_scope("hidden"):
        weights = tf.get_variable("weights", [n_input, n_hidden], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_hidden], initializer=tf.constant_initializer(0.0))
        hidden = tf.nn.relu(tf.matmul(x_input, weights) + biases)

    with tf.variable_scope("hidden2"):
        weights = tf.get_variable("weights", [n_hidden, n_hidden2], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_hidden2], initializer=tf.constant_initializer(0.0))
        hidden2 = tf.nn.relu(tf.matmul(hidden, weights) + biases)

    with tf.variable_scope("out"):
        weights = tf.get_variable("weights", [n_hidden2, n_output], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_output], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(hidden2, weights) + biases

    return output

# loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_label, name='cross_entropy_loss')
# loss = tf.reduce_mean(loss_all, name='avg_loss')

#入参5，出参2
x = tf.placeholder('float',[None,n_input])
y = tf.placeholder('float',[None,n_output])

#interface构建图，返回包含预测结果的tensor
pred = inference(x)

# 计算损失函数
# cross_entropy = tf.reduce_mean(-tf.reduce_sum(y* tf.log(pred)), name='avg_loss')
# cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=y), name='avg_loss')
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y), name='avg_loss')
# 定义优化器，梯度下降
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

#计算损失
# loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='cross_entropy_loss')
# loss = tf.reduce_mean(loss_all, name='avg_loss')

# 定义准确率计算
#equal比较两个列表返回True False列表，argmax返回最大数列表值的下标
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#tf.cast转换列表的数据类型取平均得到正确率
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

#初始化
init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    Y = tf.one_hot(Y, classes).eval()  # 在使用t.eval()时，等价于：tf.get_default_session().run(t).
    X, Y = shuffle(X, Y)

    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    x_train = X[0:3000]
    y_train = Y[0:3000]

    x_validate = X[3000:4000]
    y_validate = Y[3000:4000]

    x_test = X[4000:]
    y_test = Y[4000:]

    validate_data = {x:x_validate, y:y_validate}
    test_data = {x: x_test, y: y_test}
    for i in range(training_step):
        avg_loss = 0.
        total_batch = int(len(data.values[:, 5:10].tolist()) / batch_size)
        # xs,ys为每个batch_size的训练数据与对应的标签
        for j in range(total_batch):
            xs = x_train
            ys = y_train
            _, l = sess.run([optimizer, cross_entropy], feed_dict={x: xs, y: ys})
            avg_loss += l / total_batch
        # 每1000次训练打印一次损失值与验证准确率
        if i % 10 == 0:
            validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
            print("after %d training steps, the loss is %g, the validation accuracy is %g" % (
                i, l, validate_accuracy))

    print("the training is finish!")
    # 最终的测试准确率

    acc = sess.run(accuracy, feed_dict=test_data)
    # test_data2 = {x:[[-1.25,1.35,4.52,9.77,2.25]]}
    test_data2 = {x: [[-1.25, 1.36, 4.52, 9.76], [0.75, 5.67, 3.63, 1.64], [-0.25, 2.29, 2.98, 3.45],
                      [0.25, 4.29, 3.36, 1.87]]}  # 俄罗斯/沙特,埃及/乌拉圭,摩洛哥/伊朗，葡萄牙/西班牙
    acc2 = sess.run(pred,feed_dict=test_data2)
    # print(acc2)
    # sum = 0
    # min = abs(np.min(acc2[0]))
    # for i in acc2[0]:
    #     sum = i + min +sum
    # print(sum)
    # loss_rate = (acc2[0][0]+min)/sum
    # ping_rate = (acc2[0][1]+min)/sum
    # win_rate  = (acc2[0][2]+min)/sum
    # print(acc2[0][2]+min,loss_rate)
    print("the test accuarcy is:", acc)
    print('主队获胜概率：',acc2)

    # saver(sess,'saved_model/model.ckpt')
def getWin(x_data):
    # acc = sess.run(accuracy, feed_dict=test_data)
    # test_data2 = {x: [[-1.25, 1.35, 4.52, 9.77, 2.25]]}
    acc2 = sess.run(pred, feed_dict=x_data)
    # print(acc2)
    sum = 0
    # min = abs(np.min(acc2[0]))
    # for i in acc2[0]:
    #     sum = i + min + sum
    # print(sum)
    # loss_rate = (acc2[0][0] + min) / sum
    # ping_rate = (acc2[0][1] + min) / sum
    # win_rate = (acc2[0][2] + min) / sum
    # print(acc2[0][2] + min, loss_rate)
    # # print("the test accuarcy is:", acc)
    print('主队获胜概率：',acc2)

test_data2 = {x:[[-1.25,1.36,4.52,9.76,2.25],[0.75,5.67,3.63,1.64,2],[-0.25,2.29,2.98,3.45,1.75],[0.25,4.29,3.36,1.87,2]]}#俄罗斯/沙特,埃及/乌拉圭,摩洛哥/伊朗，葡萄牙/西班牙
# getWin(test_data2)
