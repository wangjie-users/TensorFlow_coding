#coding:utf-8
'''
    Created on 208/5/11 by Jie Wang
    用MLP实现手写字体识别

'''

import tensorflow as tf
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True)

#定义超参数
learning_rate = 0.001
epochs = 25
batch_size = 100
display_step = 1

#设置MLP模型参数，用两个隐层，每个隐层用256个节点
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784  #每张图片是28x28，故有784维
n_classes = 10 #共分为10个类别

#定义占位符
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

#初始化学习参数,用字典存储
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])), # 并说明维度
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

#创建模型
def multilayer_perception(x, weights, biases):
    #第一层隐藏层,用relu函数激活
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #第二层隐藏层
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    #输出层
    out_layer = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])
    return out_layer

# 预测输出
pred = multilayer_perception(x, weights, biases)
# 定义loss
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
#定义优化器，此处用的是Adam优化器，Adam可以自动调整学习率
optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate).minimize(cost)

#设置GPU显存计算
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        if epoch % display_step ==0:
            print("epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
    print("finished!!!")
    #测试model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))



