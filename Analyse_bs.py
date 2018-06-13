import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn import preprocessing

# sess = tf.InteractiveSession()
path = 'data/game_info.csv'
data = pd.read_csv(path)#dateframe格式
isTrain = True
# checkpoint_steps = 50
checkpoint_dir = 'E:/Anaconda3/pycharmProjects/WorldCup/temp/'

n_input = 5
# n_output = 14
#学习率
learning_rate = 0.01
training_step = 200
batch_size = 100

X = data.values[:, 5:10].tolist()
Y = data.values[:, 16:17].tolist()
print(X)
Y_temp = []
min_Y = np.min(Y)
for i in Y:
    Y_temp.append(int(i[0]))
Y = Y_temp
print (Y)
Y_max = np.max(Y)+1
# Y_max = 1
n_output = Y_max

classes = Y_max
print(Y_max)
n_hidden = 8
n_hidden2 = 8
# n_hidden3 = 8

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

    # with tf.variable_scope("hidden3"):
    #     weights = tf.get_variable("weights", [n_hidden2, n_hidden3], initializer=tf.random_normal_initializer(stddev=0.1))
    #     biases = tf.get_variable("biases", [n_hidden3], initializer=tf.constant_initializer(0.0))
    #     hidden3 = tf.nn.relu(tf.matmul(hidden2, weights) + biases)

    with tf.variable_scope("out"):
        weights = tf.get_variable("weights", [n_hidden2, n_output], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_output], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(hidden2, weights) + biases

    return output

def getrate(x_data,num):
    acc2 = sess.run(pred, feed_dict=x_data)
    # print(acc2)
    sum = 0
    for ij in range(len(acc2)):
        min = abs(np.min(acc2[ij]))
        for i in acc2[ij]:
            sum = i + min + sum
        print(sum)
        # loss_rate = (acc2[0][0]+min)/sum
        # ping_rate = (acc2[0][1]+min)/sum
        # win_rate  = (acc2[0][2]+min)/sum
        # print(acc2[0][2]+min,loss_rate)
        sort_top = sorted(acc2[ij])
        sort_top.reverse()
        sort_top3 = sort_top[0:num]
        score_type_list = []
        for i in sort_top3:
            score_type = acc2[ij].tolist().index(i)
            score_type_list.append(score_type)
        sum_t = np.sum(sort_top3)

        def getWinpercent(index_list, sum, list):
            # sum = 0
            percent_list = []
            # for i in list:
            #     if i > 0:
            #         sum = sum + i
            for index_l in index_list:
                percent = str(round(list[index_l] * 100 / sum)) + '%'
                percent_list.append(percent)
            return percent_list

        percent_list = getWinpercent(score_type_list, sum_t, acc2[ij])

        dict_result = {'1-0':1,'2-0':4,'2-1':5,'3-0':9,'3-1':10,'3-2':11,'4-0':16,'4-1':17,'4-2':18,'4-3':19,'0-0':0,'1-1':3,'2-2':8,'3-3':15,'4-4':24,'0-1':2,'0-2':6,'1-2':7,'0-3':12,'1-3':13,'2-3':14,'0-4':20,'1-4':21,'2-4':22,'3-4':23}

        score_list = []
        for (key, value) in dict_result.items():
            for score_type2 in score_type_list:
                if value == score_type2:
                    score = key
                    score_list.append(score)

        # print("the test accuarcy is:", acc)
        for i in range(num):
            score = score_list[i]
            percent = percent_list[i]
            print("%s:预测:game%s,%s,概率%s" % (i+1,ij+1, score, percent))

# loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_label, name='cross_entropy_loss')
# loss = tf.reduce_mean(loss_all, name='avg_loss')

#入参5，出参2
x = tf.placeholder('float',[None,n_input],name= 'x')
y = tf.placeholder('float',[None,n_output],name = 'y_')

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

saver = tf.train.Saver()

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
    if isTrain:
        for i in range(training_step):
            avg_loss = 0.
            total_batch = int(len(data.values[:, 5:10].tolist()) / batch_size)
            # xs,ys为每个batch_size的训练数据与对应的标签
            for j in range(total_batch):
                xs = x_train
                ys = y_train
                _, l = sess.run([optimizer, cross_entropy], feed_dict={x: xs, y: ys})
                avg_loss += l / total_batch
            # 每10次训练打印一次损失值与验证准确率
            if i % 100 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
                print("after %d training steps, the loss is %g, the validation accuracy is %g" % (
                    i, l, validate_accuracy))
                saver.save(sess,checkpoint_dir+'model_bs.ckpt',global_step=i+1)
                print('save,i')

        print("the training is finish!")
        # 最终的测试准确率
        acc = sess.run(accuracy, feed_dict=test_data)
        print("the test accuarcy is:", acc)
        test_data2 = {x: [[-1.25, 1.36, 4.52, 9.76, 2.25], [0.75, 5.67, 3.63, 1.64, 2], [-0.25, 2.29, 2.98, 3.45, 1.75],
                          [0.25, 4.29, 3.36, 1.87, 2]]}  # 俄罗斯/沙特,埃及/乌拉圭,摩洛哥/伊朗，葡萄牙/西班牙
        getrate(test_data2,0)
    else:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            graph = tf.get_default_graph()
            input = graph.get_tensor_by_name('x:0')
            output = graph.get_tensor_by_name('y_:0')
            X = [[-1.25,1.36,4.52,9.76,2.25],[0.75,5.67,3.63,1.64,2],[-0.25,2.29,2.98,3.45,1.75],[0.25,4.29,3.36,1.87,2]]
            pred = sess.run(output,feed_dict={input:X})
            print (pred)
        else:
            pass

    print("the training is finish!")
    # 最终的测试准确率
    acc = sess.run(accuracy, feed_dict=test_data)
    test_data2 = {x:[[-1.25,1.36,4.52,9.76,2.25],[0.75,5.67,3.63,1.64,2],[-0.25,2.29,2.98,3.45,1.75],[0.25,4.29,3.36,1.87,2]]}#俄罗斯/沙特,埃及/乌拉圭,摩洛哥/伊朗，葡萄牙/西班牙
    getrate(test_data2,0)
    # acc2 = sess.run(pred,feed_dict=test_data2)
    # print(acc2)
    # sum = 0
    # min = abs(np.min(acc2[0]))
    # for i in acc2[0]:
    #     sum = i + min +sum
    # print(sum)
    # # loss_rate = (acc2[0][0]+min)/sum
    # # ping_rate = (acc2[0][1]+min)/sum
    # # win_rate  = (acc2[0][2]+min)/sum
    # # print(acc2[0][2]+min,loss_rate)
    # sort_top = sorted(acc2[0])
    # sort_top.reverse()
    # sort_top3 = sort_top[0:5]
    # score_type_list = []
    # for i in sort_top3:
    #     score_type = acc2[0].tolist().index(i)
    #     score_type_list.append(score_type)
    # sum_t = np.sum(sort_top3)
    # def getWinpercent(index_list,sum,list):
    #     # sum = 0
    #     percent_list = []
    #     # for i in list:
    #     #     if i > 0:
    #     #         sum = sum + i
    #     for index_l in index_list:
    #         percent = str(round(list[index_l]*100/sum))+'%'
    #         percent_list.append(percent)
    #     return  percent_list
    #
    # percent_list = getWinpercent(score_type_list,sum_t,acc2[0])
    #
    # dict_result = {'0-0': 0, '0-1': 1, '0-2': 2, '0-3': 3, '0-4': 4, '0-5': 5, '0-6': 6, '0-7': 7, '1-0': 8, '1-1': 9,
    #                '1-2': 10, '1-3': 11, '1-4': 12, '1-5': 13, '1-6': 14, '1-7': 15, '2-0': 16, '2-1': 17, '2-2': 18,
    #                '2-3': 19, '2-4': 20, '2-5': 21, '2-6': 22, '2-7': 23, '3-0': 24, '3-1': 25, '3-2': 26, '3-3': 27,
    #                '3-4': 28, '3-5': 29, '3-6': 30, '3-7': 31, '4-0': 32, '4-1': 33, '4-2': 34, '4-3': 35, '4-4': 36,
    #                '4-5': 37, '4-6': 38, '4-7': 39, '5-0': 40, '5-1': 41, '5-2': 42, '5-3': 43, '5-4': 44, '5-5': 45,
    #                '5-6': 46, '5-7': 47, '6-0': 48, '6-1': 49, '6-2': 50, '6-3': 51, '6-4': 52, '6-5': 53, '6-6': 54,
    #                '6-7': 55, '7-0': 56, '7-1': 57, '7-2': 58, '7-3': 59, '7-4': 60, '7-5': 61, '7-6': 62, '7-7': 63}
    #
    # score_list = []
    # for (key, value) in dict_result.items():
    #     for score_type2 in score_type_list:
    #         if value == score_type2:
    #             score = key
    #             score_list.append(score)
    #
    # print("the test accuarcy is:", acc)
    # for i in range(5):
    #     score = score_list[i]
    #     percent = percent_list[i]
    #     print("%s预测:俄罗斯:沙特%s,概率%s"%(i,score,percent))
    # print('俄罗斯对阵沙特获胜概率：',round(win_rate*100,2),'平的概率：',round(ping_rate*100,2))

    # saver(sess,'saved_model/model.ckpt')


def getrate(x_data):
    acc2 = sess.run(pred, feed_dict=x_data)
    # print(acc2)
    sum = 0
    min = abs(np.min(acc2[0]))
    for i in acc2[0]:
        sum = i + min + sum
    print(sum)
    # loss_rate = (acc2[0][0]+min)/sum
    # ping_rate = (acc2[0][1]+min)/sum
    # win_rate  = (acc2[0][2]+min)/sum
    # print(acc2[0][2]+min,loss_rate)
    sort_top = sorted(acc2[0])
    sort_top.reverse()
    sort_top3 = sort_top[0:5]
    score_type_list = []
    for i in sort_top3:
        score_type = acc2[0].tolist().index(i)
        score_type_list.append(score_type)
    sum_t = np.sum(sort_top3)

    def getWinpercent(index_list, sum, list):
        # sum = 0
        percent_list = []
        # for i in list:
        #     if i > 0:
        #         sum = sum + i
        for index_l in index_list:
            percent = str(round(list[index_l] * 100 / sum)) + '%'
            percent_list.append(percent)
        return percent_list

    percent_list = getWinpercent(score_type_list, sum_t, acc2[0])

    dict_result = {'0-0': 0, '0-1': 1, '0-2': 2, '0-3': 3, '0-4': 4, '0-5': 5, '0-6': 6, '0-7': 7, '1-0': 8, '1-1': 9,
                   '1-2': 10, '1-3': 11, '1-4': 12, '1-5': 13, '1-6': 14, '1-7': 15, '2-0': 16, '2-1': 17, '2-2': 18,
                   '2-3': 19, '2-4': 20, '2-5': 21, '2-6': 22, '2-7': 23, '3-0': 24, '3-1': 25, '3-2': 26, '3-3': 27,
                   '3-4': 28, '3-5': 29, '3-6': 30, '3-7': 31, '4-0': 32, '4-1': 33, '4-2': 34, '4-3': 35, '4-4': 36,
                   '4-5': 37, '4-6': 38, '4-7': 39, '5-0': 40, '5-1': 41, '5-2': 42, '5-3': 43, '5-4': 44, '5-5': 45,
                   '5-6': 46, '5-7': 47, '6-0': 48, '6-1': 49, '6-2': 50, '6-3': 51, '6-4': 52, '6-5': 53, '6-6': 54,
                   '6-7': 55, '7-0': 56, '7-1': 57, '7-2': 58, '7-3': 59, '7-4': 60, '7-5': 61, '7-6': 62, '7-7': 63}

    score_list = []
    for (key, value) in dict_result.items():
        for score_type2 in score_type_list:
            if value == score_type2:
                score = key
                score_list.append(score)

    # print("the test accuarcy is:", acc)
    for i in range(5):
        score = score_list[i]
        percent = percent_list[i]
        print("%s预测:俄罗斯:沙特%s,概率%s" % (i, score, percent))