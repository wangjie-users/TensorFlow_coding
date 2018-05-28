'''
    用 Softmax Regression 做多分类问题： y = Softmax(Wx +b)
    Created on 2018/5/9
    @author: Jie Wang

'''

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("mnist_data/", one_hot = True) #准备训练数据
print(mnist.train.images.shape) #显示训练集的维度：（55000,784）

#定义占位符，表示训练集数据，train_x和train_y
x = tf.placeholder(tf.float32, [None, 784]) #None表示任意维度，即输入x的数量任意，但每个x都是784维，因为每张图是784维
y = tf.placeholder(tf.float32, [None, 10])

#初始化学习参数
w = tf.Variable(tf.random_normal(([784, 10]))) #由矩阵乘法可得出w的维度，一般将w设为一个服从正态分布的随机值
b = tf.Variable(tf.zeros([10])) #将b初始化为0

#创建模型
pred = tf.nn.softmax(tf.matmul(x, w) + b)

#定义交叉熵损失函数，再对每个batch的结果取均值
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1)) #参数表示按列相加
# cost = tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y) #也可直接调用函数

#使用梯度下降优化器,并设置学习率
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

#下面开始训练模型并输出中间状态参数
training_epochs = 25 #将整个训练集迭代25次
batch_size = 100 #训练样本总数为55000，batch_size=100代表在训练过程中一次取100条数据进行训练
display_step = 1 #表示每训练一次就把具体的中间状态显示出来
saver = tf.train.Saver() #用于保存模型
model_path = "model/521model.ckpt" #模型存放路径

#设置显存
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #常规操作，初始化全局变量
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size) # 55000/100，训练完所有样本集一次需要550次迭代
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size) #准备训练数据train_x和train_y
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            c = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += c / total_batch #计算平均loss值
        if (epoch+1) % display_step == 0:
            print("epoch:", '%04d' % (epoch+1), "cost=","{:.9f}".format(avg_cost))
    print("finished!")

    #测试model
    #测试准确率的做法：直接判断预测的结果和真实的标签是否相同，然后将正确的个数除以总个数，算出正确率
    # tf.argmax表示从一个tensor中寻找最大值的序号，tf.equal用来判断预测的数字类别是否就是正确的类别
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

    #tf.cast将之前的bool值转化为float32
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #用eval转化为字典形式的字符串输出，类似于run，需要喂入数据
    # print("accuracy:", accuracy.eval({x:mnist.test.images, y: mnist.test.labels}))
    print("accuracy:", sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
    # #保存模型
    # save_path = saver.save(sess, model_path)
    # print("model saved in files: %s" % save_path)

#使用模型，新建一个session，调用存储的模型
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     saver.restore(sess, model_path) #恢复模型变量，w和b
#
#     #测试model是否正确导入
#     correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
#     accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#     print("accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
#
#     #输入新数据
#     output = tf.argmax(pred, 1) #把模型得到的one-hot编码转化为数字，作为最终的输出
#     batch_xs, batch_ys = mnist.train.next_batch(2)
#     outputval, predv = sess.run([output, pred], feed_dict={x:batch_xs})
#     print(outputval, predv,batch_ys) # outptval表示预测结果，predv表示模型的真实输出值（未转化成数字），batch_ys是真实数字的one-hot编码























