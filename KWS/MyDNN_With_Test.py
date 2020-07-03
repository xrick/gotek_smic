#tensorflow 1.5
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
import scipy.io as spio
import matplotlib.pyplot as plt
tf.disable_v2_behavior()
# tf.reset_default_graph()
# tf.reset_default_graph()
# old_v = tf.logging.get_verbosity()
# tf.logging.set_verbosity(tf.logging.ERROR)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def runningMeanFast(x, N):
    return np.convolve(x, np.ones((N,))/N)[(N-1):]

def datachange(input):
    out = []
    for i in range(len(input)):
        if input[i]==1:
            out.append([1,0,0])
        elif input[i]==2:
            out.append([0,1,0])
        else:
            out.append([0,0,1])
    return out

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]

    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

train_data = spio.loadmat('./MyTrainData/kws_train_data_20200703.mat')
train_lbl = spio.loadmat('./MyTrainData/kws_train_lbl_20200703.mat')
x_train = train_data['xtrain']
y_train = train_lbl['ytrain'][0]
y_train = datachange(y_train)

test_data = spio.loadmat('./MyTrainData/kws_test_data_20200703.mat')
test_lbl = spio.loadmat('./MyTrainData/kws_test_lbl_20200703.mat')
x_test = test_data['xtrain']
y_test = test_lbl['ytrain'][0]
y_test = datachange(y_test)

# test_data = spio.loadmat('train_data/test.mat')
# x_test = test_data['x_test']
# y_test = test_data['y_test']

# y_train = datachange(y_train)
# y_test = datachange(y_test)

# Parameters
learning_rate = 0.001
batch_size = 32
n_band = 1600#40*41 # mfcc dim * context size
#n_band = 10*41 # filter dim * context size (i)
#n_band = 40*41 # filter dim * context size (ii) (iii)
n_hidden1 = 128
n_hidden2 = 128
n_hidden3 = 128
n_classes = 3
optimizer_name = "Adam"
# tf Graph input
x = tf.placeholder(tf.float32, [None, n_band])
# y = tf.placeholder(tf.float32, [None, n_classes])
y = tf.placeholder('float')

##define network
def dnn128(input_data):
    weights = {
        'hidden1': tf.Variable(tf.random.normal([n_band, n_hidden1], dtype=tf.float32, stddev=0.1)),
        'hidden2': tf.Variable(tf.random.normal([n_hidden1, n_hidden2], dtype=tf.float32, stddev=0.1)),
        'hidden3': tf.Variable(tf.random.normal([n_hidden2, n_hidden3], dtype=tf.float32, stddev=0.1)),
        'out': tf.Variable(tf.random.normal([n_hidden3, n_classes], dtype=tf.float32, stddev=0.1))
    }
    biases = {
        'hidden1': tf.Variable(tf.zeros([n_hidden1], dtype=tf.float32)),
        'hidden2': tf.Variable(tf.zeros([n_hidden2], dtype=tf.float32)),
        'hidden3': tf.Variable(tf.zeros([n_hidden3], dtype=tf.float32)),
        'out': tf.Variable(tf.zeros([n_classes], dtype=tf.float32))
    }

    x1 = tf.nn.relu(tf.add(tf.matmul(input_data, weights['hidden1']), biases['hidden1']))
    # x1=tf.nn.dropout(x1,0.95)
    x2 = tf.nn.relu(tf.add(tf.matmul(x1, weights['hidden2']), biases['hidden2']))
    # x2=tf.nn.dropout(x2,0.95)
    x3 = tf.nn.relu(tf.add(tf.matmul(x2, weights['hidden3']), biases['hidden3']))
    # x2=tf.nn.dropout(x2,0.95)
    logits = tf.matmul(x3, weights['out']) + biases['out']
    return logits

prediction = dnn128(x)
# loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=prediction, labels=y))
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)  #training logic

error_train=[]
error_test=[]
weight1 = []
weight2 = []
weight3 = []
weight4 = []
bias1 = []
bias2 = []
bias3 = []
bias4 = []
init = tf.global_variables_initializer()

'''Add ops to save and restore all the variables.'''
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)

    '''    Restore variables from disk.    '''
#    checkpoint = tf.train.latest_checkpoint("checkpoints/checkpoints_1017_256")
#    saver.restore(sess, checkpoint)
    epochnum=501
    epoch_loss = []
    loss_value = 0.0
    for epoch in range(epochnum):
#        for step in range(60):
        for step in range(100):
            x_n, y_n = next_batch(batch_size, x_train, y_train)
            # x_t, y_t = next_batch(batch_size, x_test, y_test)
            sess.run(train_step, feed_dict={x: x_n, y: y_n})
            loss_value = sess.run(loss, feed_dict={x: x_test, y: y_test})
            epoch_loss.append(loss_value)
            print('Epoch', epoch, 'completed out of', step, 'loss:', loss_value)

        # float_logit = tf.cast(tf.argmax(prediction, 1), tf.float32)
        # float_y = tf.cast(y, tf.float32)
        correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #calculating the accuracy
        acc = accuracy.eval(feed_dict={x: x_train, y: y_train})
        print('Accuracy:', acc)
        # print('Step {}: rate {}, accuracy {}, Loss {}'.format(epoch, learning_rate, acc * 100, loss_value))
        # plt.subplot(1, 2, 1)
        # plt.plot(loss_value)
        # plt.title('Epoch Loss')
        # plt.show()
        # acc1 = sess.run(accuracy, feed_dict={x: x_test, y: y_test})
            # spio.savemat('weight/w_3layer128.mat', {'w1': weight1,'w2':weight2,'w3':weight3,'w4':weight4,
            #                                         'b1':bias1,'b2':bias2,'b3':bias3,'b4':bias4})

    '''    # Save the variables to disk. '''
    #save_path = saver.save(sess, "checkpoints/checkpoints_1211/model_1211d_ep_{}.ckpt".format(epoch))


print("training finished!")
# plt.plot(range(len(error_train)), error_train, 'b', label='Training accuracy')
# plt.plot(range(len(error_test)), error_test, 'r', label='Test accuracy')
# plt.title('accuracy')
# plt.xlabel('epoch',fontsize=16)
# plt.ylabel('accuracy',fontsize=16)
# plt.legend()
# plt.figure()
# plt.show()
#
#
# a = runningMeanFast(error_train, 50)
# b = runningMeanFast(error_test, 50)
# plt.plot(range(len(a)), a, 'b', label="train")
# plt.plot(range(len(b)), b, 'r', label="test")
# plt.grid()
# plt.xlabel('epoch',fontsize=16)
# plt.ylabel('accuracy',fontsize=16)
# plt.xlim(0,500)
# plt.legend()
# plt.figure()
# plt.show()

