'''
<<<<<<< HEAD
created in 2018/4/15
linear regeression using tensorflow
=======
created on 2018/4/15
logistic regeression using tensorflow
>>>>>>> 6458db65fdf781775ad84872673a0000682d3035
@author: Jie Wang
LinearRegression Algorithms

'''

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#准备训练集training data
train_data = pd.read_csv("ex1data1.txt",names = ['population','profit'])
# print(train_data.head())
train_X = np.asarray([train_data.population.T]).transpose()
train_Y = np.asarray([train_data.profit]).transpose()
n_sample = train_X.shape[0]
# print(train_X,train_Y)
# print(n_sample)

#定义输入变量及参数
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable(np.random.randn())
b = tf.Variable(np.random.rand())

#建立模型
preb = tf.add(tf.multiply(x,w),b)
#定义损失函数
cost = tf.reduce_sum(tf.pow(preb - y,2) / (2 * n_sample))
#梯度下降求参数
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
#定义一个全局初始化变量
init = tf.global_variables_initializer()

#训练模型，建立会话并运行
training_epochs = 1000 #定义训练次数
# 设置显存计算
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        for (X, Y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={x:X, y:Y}) #喂入训练数据

        #显示训练日志文件
        display_steps = 50
        if (epoch + 1) % display_steps == 0:
            c = sess.run(cost, feed_dict={x:X, y:Y})
            print("epoch:{}".format(epoch+1), "cost ={}".format(c), "w=", sess.run(w), "b=", sess.run(b) )

    print("optimization finished!")
    training_cost = sess.run(cost,feed_dict={x:X, y:Y})
    print("training_cost=", training_cost,"w=", sess.run(w),"b=", sess.run(b))

    #可视化，这部分也必须写到with内容中
    plt.plot(train_X, train_Y, 'ro', label = 'original data')
    plt.plot(train_X, sess.run(w) * train_X + sess.run(b), label = "fitted line")
    plt.legend()
    plt.show()








