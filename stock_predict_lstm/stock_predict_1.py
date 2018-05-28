# coding:utf-8
'''
Created on 2018/5/8
@author : Jie Wang
用LSTM预测股价走势，只含一个特征的情况；
本例取每日最高价作为输入特征，后一天的最高价作为标签；

'''
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers


# ——————————导入数据——————————

df = pd.read_csv("dataset_1.csv", index_col=0)
data = np.array(df['最高价']) # 获取最高价那一列
data = data[::-1] # 逆序操作，使数据按照日期先后顺序排列
normalize_data = (data - np.mean(data)) / np.std(data) # 标准化
normalize_data = normalize_data[:, np.newaxis] # 增加一个维度用于存放label

# ——————————生成训练集和测试集——————————
time_step = 20
n_hidden = 10
batch_size = 60
input_size = 1
output_size = 1
learning_rate = 0.001
trainning_steps = 1000
display_step = 10


train_x = []
train_y = []
test_x = []
test_y = []

for i in range(len(normalize_data) - time_step - 500):
    x= normalize_data[i:i+time_step]
    y = normalize_data[i+1:i+time_step+1]
    # print(x)
    train_x.append(x.tolist())
    train_y.append(y.tolist())

for j in range(500 - time_step -1):
    x_ = normalize_data[i+1+j : i+1+j+time_step]
    y_ = normalize_data[i+1+j+1 : i+1+j+time_step+1]
    test_x.append(x_.tolist())
    test_y.append(y_.tolist())

# # ——————————构造LSTM——————————
X = tf.placeholder(tf.float32, [None, time_step, input_size])
Y = tf.placeholder(tf.float32, [None, time_step, output_size])
x1 = tf.unstack(X, time_step, 1)
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0)
outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)
pred = layers.fully_connected(outputs[-1], output_size, activation_fn=None)

# ——————————训练模型——————————
# 定义损失函数
cost = tf.reduce_mean(tf.square(pred - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(trainning_steps):
        # batch_x = tf.reshape(train_x, [batch_size, time_step, input_size])
        # batch_y = tf.reshape(train_y, [batch_size, time_step, output_size])
        sess.run(optimizer, feed_dict={X:train_x, Y:train_y})
        if step % display_step == 0 or step == 1:
            loss = sess.run(cost, feed_dict={X:train_x, Y:train_y})
            print("step:", '%04d' % (step), "cost=""{:.9f}".format(loss))
    print("finished!!!")





















