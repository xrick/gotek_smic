#tensorflow 1.5
import numpy as np
import tensorflow as tf
# import tensorflow.compat.v1 as tf
import scipy.io as spio
import matplotlib.pyplot as plt
# tf.disable_v2_behavior()
# tf.reset_default_graph()
# tf.reset_default_graph()

train_data = spio.loadmat('./MyTrainData/kws_train_data_20200703.mat')
train_lbl = spio.loadmat('./MyTrainData/kws_train_lbl_20200703.mat')
x_train = train_data['xtrain']
y_train = train_lbl['ytrain'][0]




# test_data = spio.loadmat('train_data/test.mat')
# x_test = test_data['x_test']
# y_test = test_data['y_test']

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

    
# y_train = datachange(y_train)
# y_test = datachange(y_test)

# Parameters
learning_rate = 0.01
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
y = tf.placeholder(tf.int64, [None])


weights = {
    'hidden1': tf.Variable(tf.random_normal([n_band, n_hidden1],dtype=tf.float32,stddev=0.1)),
    'hidden2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2],dtype=tf.float32,stddev=0.1)),
    'hidden3': tf.Variable(tf.random_normal([n_hidden2, n_hidden3],dtype=tf.float32,stddev=0.1)),
    'out': tf.Variable(tf.random_normal([n_hidden3, n_classes],dtype=tf.float32,stddev=0.1))
}
biases = {
    'hidden1': tf.Variable(tf.zeros([n_hidden1],dtype=tf.float32)),
    'hidden2': tf.Variable(tf.zeros([n_hidden2],dtype=tf.float32)),
    'hidden3': tf.Variable(tf.zeros([n_hidden3],dtype=tf.float32)),
    'out': tf.Variable(tf.zeros([n_classes],dtype=tf.float32))
}

##train
x1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']), biases['hidden1']))
# x1=tf.nn.dropout(x1,0.95)
x2 = tf.nn.relu(tf.add(tf.matmul(x1, weights['hidden2']), biases['hidden2']))
# x2=tf.nn.dropout(x2,0.95)
x3 = tf.nn.relu(tf.add(tf.matmul(x2, weights['hidden3']), biases['hidden3']))
# x2=tf.nn.dropout(x2,0.95)
logits = tf.matmul(x3, weights['out']) + biases['out']
loss = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=y))
train_step = tf.contrib.layers.OPTIMIZER_CLS_NAMES[optimizer_name](learning_rate).minimize(loss) # training logic
float_logit = tf.cast(tf.argmax(logits, 1), tf.float32)
float_y = tf.cast(y, tf.float32)
print("logit is {} and y is {}".format(float_logit, float_y))
correct_prediction = tf.equal(float_logit, float_y)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) #calculating the accuracy

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
    for epoch in range(epochnum):
#        for step in range(60):
        for step in range(100):
            x_n, y_n = next_batch(batch_size, x_train, y_train)
            sess.run(train_step, feed_dict={x: x_n, y: y_n})
        loss_value = sess.run(loss, feed_dict={x: x_n, y: y_n})
        acc = sess.run(accuracy, feed_dict={x: x_n, y: y_n})

        print('Step {}: rate {}, accuracy {}, Loss {}'.format(epoch, learning_rate, acc * 100, loss_value))
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

