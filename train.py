import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data
import pandas as pd

old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

learning_rate = 0.1
trainig_step = 30000
batch_size = 100

#球队A，球队B，主客场，A球队世界排名，B球队世界排名,[初步想法根据比分给做出一个预测盘口如果]
#[受让两球，受让两球/球半 受让球半 受让球半/ 一球 受让一球 受让一球/球半 受让半球，受让半球/平手，平手，平手/让半球 让半球，让一球，让一球半，让两球，让两球半]
#例如1：1  0.6 A 0.9 B 1.4
n_input = 5
n_hidden = 6
n_hidden2 = 3
n_labels = 2

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
        weights = tf.get_variable("weights", [n_hidden2, n_labels], initializer=tf.random_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [n_labels], initializer=tf.constant_initializer(0.0))
        output = tf.matmul(hidden2, weights) + biases

    return output

def read_data(file_name):
    filename_queue = tf.train.string_input_producer(file_name)

    reader = tf.TextLineReader()
    key, value = reader.read(filename_queue)

    # Default values, in case of empty columns. Also specifies the type of the
    # decoded result.
    record_defaults = [[1],[''], [''], [''], [''], [1.0],[1.0],[1.0],[1.0], [1.0],[1], [1],[''],[''],[1.0],[1.0]]
    # record_defaults = [[1], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''], [''],
    #                    ['']]
    index,col1, col2, col3, col4,col5,col6,col7,col8, col9, col10, col11, col12,col13,col14,col15= tf.decode_csv(
        value, record_defaults=record_defaults)
    features = tf.stack([col5,col6, col7, col8,col9],axis=0)
    values = tf.stack([col14,col15],axis=0)

    return  features,values
    # with tf.Session() as sess:
    #     # Start populating the filename queue.
    #     coord = tf.train.Coordinator()
    #     threads = tf.train.start_queue_runners(coord=coord)
    #
    #     for i in range(5000):
    #         # Retrieve a single instance:
    #         example, label = sess.run([features, values])
    #         print(example,label)
    #     coord.request_stop()
    #     coord.join(threads)


def train(x2,y2,x1,y1):
    x = tf.placeholder("float", [None,n_input])
    y = tf.placeholder("float",  [None,n_labels])
    pred = inference(x)

    features = x2
    values = y2
    features2 = x1
    values2 = y2

    # # 计算损失函数
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    # # 定义优化器
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    loss_all = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y, name='cross_entropy_loss')
    loss = tf.reduce_mean(loss_all, name='avg_loss')
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    # 定义准确率计算
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        #定义验证集与测试集
        validate_data = {x: features, y: values}
        test_data = {x: features2,y: values2}

        for i in range(trainig_step):
            avg_loss = 0.
            total_batch = int(len(features)/ batch_size)
            # xs,ys为每个batch_size的训练数据与对应的标签
            for j in range(total_batch):
                xs = features
                ys = values
                _, l = sess.run([optimizer, loss], feed_dict={x:xs, y: ys})
                avg_loss += l / total_batch
            # 每1000次训练打印一次损失值与验证准确率
            if i % 10 == 0:
                validate_accuracy = sess.run(accuracy, feed_dict=validate_data)
                print("after %d training steps, the loss is %g, the validation accuracy is %g" % (
                i, l, validate_accuracy))

        print("the training is finish!")
        # 最终的测试准确率
        acc = sess.run(accuracy, feed_dict=test_data)
        print("the test accuarcy is:", acc)

if __name__ == "__main__":
    # read_data(path)
    # print(read_data(path),type(read_data(path)))
    # train(read_data(path),read_data(path2))

    path = 'data/game_info.csv'
    path2 = 'data/game_info_test.csv'
    data = pd.read_csv(path)
    x = data.values[:,5:10].tolist()
    y = data.values[:,14:16].tolist()

    data2 = pd.read_csv(path2)
    x2 = data2.values[:, 5:10].tolist()
    y2 = data2.values[:, 14:16].tolist()


    train(x,y,x2,y2)