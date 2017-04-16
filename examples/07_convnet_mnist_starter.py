""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""
from __future__ import print_function

import os
import time 

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 50

# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

dropout = tf.placeholder(tf.float32, name='dropout')

# Step 4 + 5: create weights + do inference
# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

with tf.variable_scope('conv1') as scope:
    # first, reshape the image to [BATCH_SIZE, 28, 28, 1] to make it work with tf.nn.conv2d
    # use the dynamic dimension -1
    # 参数设置为 1 的时候，会自动计算大小，只能有一个地方设置为 -1
    images = tf.reshape(X,shape=[-1,28,28,1])

    # create kernel variable of dimension [5, 5, 1, 32]
    # use tf.truncated_normal_initializer()
    #tf.random_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
    #tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32, seed=None, name=None)
    #这几个都是用于生成随机数tensor的。尺寸是shape
    #random_normal: 正太分布随机数，均值mean, 标准差stddev
    #truncated_normal:截断正态分布随机数，均值mean, 标准差stddev, 不过只保留[mean - 2 * stddev, mean + 2 * stddev]
    #范围内的随机数
    #random_uniform:均匀分布随机数，范围为[minval, maxval]
    kernel = tf.get_variable('kernel',[5,5,1,32],
                    initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable(('biases',[32]),
                    initializer=tf.truncated_normal_initializer())
    #input: batchsize,长，宽，输入值的channel
    #filter：长和宽，输入的 featuremap，输出的featuremap，stride是，指的是移动的大小
    #(W−F + 2P) / S + 1
    #W: input
    #width
    #F: filter
    #width
    #P: padding
    #S: stride
    conv = tf.nn.conv2d(images,kernel,strides=[1,1,1,1],padding='SAME')
    covn1 = tf.nn.relu(conv+biases,name=scope.name)
    # output is of dimension BATCH_SIZE x 28 x 28 x 32

with tf.variable_scope('pool1') as scope:
    # apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding 'SAME'
    
    pool1 = tf.nn.max_pool(covn1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

    # output is of dimension BATCH_SIZE x 14 x 14 x 32

with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel = tf.get_variable('kernels', [5, 5, 32, 64], 
                        initializer=tf.truncated_normal_initializer())
    biases = tf.get_variable('biases', [64],
                        initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases, name=scope.name)

    # output is of dimension BATCH_SIZE x 14 x 14 x 64

with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                            padding='SAME')

    # output is of dimension BATCH_SIZE x 7 x 7 x 64

# 全连接层 ，输出1024个神经元
with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * 64
    
    # create weights and biases

    w = tf.get_variable('weights', [input_features, 1024],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [1024],
                        initializer=tf.random_normal_initializer())

    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])

    # apply relu on matmul of pool2 and w + b

    pool2 = tf.reshape(pool2, [-1, input_features])
    fc = tf.nn.relu(tf.matmul(pool2, w) + b, name='relu')
    # dropout，每轮结束舍弃掉一部分数据，避免 overfitting
    # apply dropout
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')
# 输出层
with tf.variable_scope('softmax_linear') as scope:
    # this you should know. get logits without softmax
    # you need to create weights and biases

    w = tf.get_variable('weights', [1024, N_CLASSES],
                        initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', [N_CLASSES],
                        initializer=tf.random_normal_initializer())
    logits = tf.matmul(fc, w) + b

# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
with tf.name_scope('loss'):
    # you should know how to do this too

    entropy = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)

    loss = tf.reduce_mean(entropy, name='loss')

# Step 7: define training op
# using gradient descent with learning rate of LEARNING_RATE to minimize cost
# don't forgot to pass in global_step

optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss,
                                        global_step=global_step)

#下面的注释是官网的代码，使用 accuracy.eval() 和optimizer.run()，减少了代码量
#cross_entropy = tf.reduce_mean(
#    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
#train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#    for i in range(20000):
#        batch = mnist.train.next_batch(50)
#       if i % 100 == 0:
#           train_accuracy = accuracy.eval(feed_dict={
#                x: batch[0], y_: batch[1], keep_prob: 1.0})
#            print('step %d, training accuracy %g' % (i, train_accuracy))
#        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
    ##### You have to create folders to store checkpoints
    # 这里有点问题，没有这个目录会报错，正常情况不会这么用
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    # 初始化的作用，intial_step = 0
    initial_step = global_step.eval()

    start_time = time.time()

    #num_example 的大小是个常量，10000
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    # 一个n_batches就是一轮，这里训练N_EPOCHS
    for index in range(initial_step, n_batches * N_EPOCHS): # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch = sess.run([optimizer, loss], 
                                feed_dict={X: X_batch, Y:Y_batch, dropout: DROPOUT}) 
        total_loss += loss_batch
        # 每 SKIP_STEP 打印一次结果，相当于 monitor 的作用，输出的是10次的平均loss
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)
    
    print("Optimization Finished!") # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))
    
    # test the model
    n_batches = int(mnist.test.num_examples/BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits], 
                                        feed_dict={X: X_batch, Y:Y_batch, dropout: DROPOUT}) 
        preds = tf.nn.softmax(logits_batch)
        #[True, False, True, True]会转换成[1, 0, 1, 1]，其平均值0.75代表了准确比例
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)   
    
    print("Accuracy {0}".format(total_correct_preds/mnist.test.num_examples))