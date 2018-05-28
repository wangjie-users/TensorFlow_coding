#coding:utf-8
'''
Created on 2018/5/12 by Jie Wang
RNN之于MNIST，是把mnist分成28x28数据。即每行28个像素，共有28行。站在时间序列的角度，其实图片没有序列概念，
但是我们可以这样理解，每一行相对于下一行是有位置关系的，不能更改顺序。比如，一个手写 “7”字，如果把28行
的上下行顺序打乱， 那么7 上面的一横就可能在中间位置，也可能在下面的位置。这样，最终的结果就不应该是7。
所以mnist的28x28可以理解为有时序关系的数据。
故每次取该batch中所有图片的一行作为一个时间序列输入



'''
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import layers



from  tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("data/", one_hot=True) #(55000,784)

#定义模型参数
n_input = 28    # 一行有28个元素，作为输入
time_steps = 28 # 序列个数，即timestep，有28行
n_hidden = 128  # 隐层block数目
n_classes = 10

# 定义占位符
x = tf.placeholder('float', [None, time_steps, n_input]) # 三维张量
y = tf.placeholder("float", [None, n_classes])

# 开始创建LSTM网络
# 在输入之前，一定要将张量转化为list，通过unstack函数
x1 = tf.unstack(x, time_steps, 1) #unstack矩阵分解函数，将原始的输入28x28调整成具有28个元素的list,每个元素是1x28的数组
# 创建TensorFlow中的cell类
lstm_cell = rnn.BasicLSTMCell(n_hidden, forget_bias=1.0) #forget_bias表示forget门的偏置，常设为1.0

# 定义好cell类以后，用静态RNN方法构成网络。前两个是必传参数：生成好的cell和输入数据
# 输入数据一定是个list或者二维张量，list的顺序就是时间序列，其中每个元素就是每个序列的值
# LSTM有两个返回值，隐层的结果和状态
outputs, states = rnn.static_rnn(lstm_cell, x1, dtype=tf.float32)

# 隐层的结果是一个list，输入是多少个时序，list就会输出多少个元素，我们只取最后一个参与后边的运算
pred = layers.fully_connected(outputs[-1], n_classes, activation_fn=None)


# 定义训练参数
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 10

# 定义损失函数和优化器
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#定义准确率
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# 设置GPU显存计算
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options =gpu_options))

# 开始训练
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(training_steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, time_steps, n_input)) # 将训练数据变成x的样子，每次训练一个batch中的所有样本
        sess.run(optimizer, feed_dict={x: batch_x, y:batch_y})
        if step % display_step == 0 or step == 1:
            acc = sess.run(accuracy, feed_dict={x:batch_x, y:batch_y})
            loss = sess.run(cost, feed_dict={x:batch_x, y:batch_y})
            print("step: " '%04d' % (step),"cost=""{:.9f}".format(loss), "accuracy={:.5f}".format(acc))
    print("finished!!!")

    #测试model,准备测试数据
    test_len = 128 #选取一个batch-size的数据作测试
    # test_data = mnist.test.images[:test_len].reshape((-1, time_steps, n_input)) # -1表示函数自己判断多少行
    test_data = tf.reshape(mnist.test.images[:, test_len], (-1, time_steps, n_input))
    # print(mnist.test.images[:test_len]) # (128,784)
    # print(test_len) #(128,28,28)
    test_label = mnist.test.labels[:test_len]
    print("accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label}))












