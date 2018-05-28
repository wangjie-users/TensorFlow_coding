#coding:utf-8
'''
构建动态RNN

'''
import tensorflow as tf
from tensorflow.contrib import rnn
from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True) #(55000,784)

n_input = 28   # 输入一行，一行有28个数据
max_time = 28  # 每张图片共有28行
lstm_size = 100 # 隐层单元（block）个数
n_classes = 10
learning_rate = 0.001
batch_size = 50  # 每批次处理50个样本
n_batch = mnist.train.num_examples / batch_size
training_steps = 5
display_step = 10

x = tf.placeholder(tf.float32, [None, 784])  # None表示第一个维度是任意的
y = tf.placeholder(tf.float32, [None, 10])

# 初始化权重(只设置了隐藏层到输出层的参数)
weights = tf.Variable(tf.truncated_normal([lstm_size, n_classes], stddev=0.1)) # truncated_normal表示截断正态分布，用于限制取值范围
biases = tf.Variable(tf.constant(0.1, shape=[n_classes]))

def lstm(X, weithts, biases):
    inputs = tf.reshape(X, [-1, max_time, n_input]) # LSTM的输入必须是三维张量[batch_size,max_time,n_input]
    lstm_cell = rnn.BasicLSTMCell(lstm_size)
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, inputs, dtype=tf.float32)
    results = tf.nn.softmax(tf.matmul(final_state[1], weights) + biases) # final_state[1]是hidden_state
    return results

pred = lstm(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(pred, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options =gpu_options))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_steps):
        for batch in range(n_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x:batch_xs, y:batch_ys})
        if step % display_step ==0 or step == 1:
            acc = sess.run(accuracy, feed_dict={x:batch_xs, y:batch_ys})
            print("step:" + str(step) + ",Testing accuracy = " + str(acc))
    print("finished!")












