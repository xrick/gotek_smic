import numpy as np
import scipy.io as spio
import timeit
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
import time
# from . import Parameters
import os
from os import path
import psutil
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()



#*********************************************************************************************************************
                                       #  Functions of NN  #
#*********************************************************************************************************************
def relu(x):
    return np.maximum(0, x)

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def runMeanFast(x, N):
    return np.convolve(x, np.ones((N,)) / N)[(N - 1):]

def datachange(input):
    out = []
    for i in range(len(input)):
        if input[i] == 0:
            out.append([1, 0])
        else:
            out.append([0, 1])
    return out

def next_batch(num, data, labels):
    idx = np.arange(0, len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[i] for i in idx]
    labels_shuffle = [labels[i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def trainProcessEntry(h1_layer_num=512, h2_layer_num=32, batch_size = 40, learning_rate=0.008, \
                      file_handle=None):
    # current_lr = learning_rate#learning_rates_list[idx]
    current_lr_str = str(learning_rate).replace(".", "_")
    file_handle.write("*********** Start Of Training Of Learning Rate {} **********\n".format(current_lr_str))
    file_handle.write("Experiment Start Time: {}\n".format(datetime.now()))
    train_all = spio.loadmat('../train_data/8+2band(25ms)/train_1106a_sharp_12.mat')
    x_train = train_all['x_data']
    train_label = spio.loadmat('../train_label/8+2band(25ms)/label_1106a_sharp_12.mat')
    y_train = train_label['y_data']
    y_train = y_train[0]

    y_train = datachange(y_train)
    # Parameters
    learning_rate = learning_rate#current_lr#learning_rates_list[idx]  #0.01
    batch_size = batch_size#128
    n_band = 40
    n_hidden1 = h1_layer_num#512
    n_hidden2 = h2_layer_num#32
    n_classes = 2
    EPOCHES = 20000#16001

    # initial the tf Graph input
    x = tf.placeholder(tf.float32, [None, n_band])
    y = tf.placeholder(tf.float32, [None, n_classes])

    # initial the
    # weights
    weights = {
        'hidden1': tf.Variable(tf.random_normal([n_band, n_hidden1], dtype=tf.float32, stddev=0.1)),
        'hidden2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2], dtype=tf.float32, stddev=0.1)),
        'out': tf.Variable(tf.random_normal([n_hidden2, n_classes], dtype=tf.float32, stddev=0.1))
    }
    biases = {
        'hidden1': tf.Variable(tf.zeros([n_hidden1], dtype=tf.float32)),
        'hidden2': tf.Variable(tf.zeros([n_hidden2], dtype=tf.float32)),
        'out': tf.Variable(tf.zeros([n_classes], dtype=tf.float32))
    }

    # define training computation procedure
    x1 = tf.nn.relu(tf.add(tf.matmul(x, weights['hidden1']),biases['hidden1']))
    x2 = tf.nn.relu(tf.add(tf.matmul(x1, weights['hidden2']),biases['hidden2']))
    pred = tf.add(tf.matmul(x2,weights['out']),biases['out'])
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred,labels=y))
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    init = tf.global_variables_initializer()

    error_train = []
    error_test = []
    weight1 = []
    weight2 = []
    weight3 = []
    bias1 = []
    bias2 = []
    bias3 = []
    can_write_flag = 1000
    incresement = 0
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        # initial used data structures
        local_start = timeit.default_timer()
        for epoch in range(EPOCHES):
            for step in range(100):
                x_n, y_n = next_batch(batch_size, x_train, y_train)
                sess.run(train_step, feed_dict={x: x_n, y: y_n})
                incresement += 1

            acc = sess.run(accuracy, feed_dict={x: x_train, y: y_train})
            if incresement % can_write_flag == 0:
                print("epoch", epoch)
                print("train : ", acc)
                local_end = timeit.default_timer()
                file_handle.write("elapsed_time:{};at_epoach:{}\n\n".format(local_end-local_start, epoch))
                file_handle.write("accurancy:{};at_epoach:{}\n\n".format(acc, epoch))
                file_handle.flush()

            error_train.append(acc)

        weight1 = weights['hidden1'].eval(sess)
        weight2 = weights['hidden2'].eval(sess)
        weight3 = weights['out'].eval(sess)
        bias1 = biases['hidden1'].eval(sess)
        bias2 = biases['hidden2'].eval(sess)
        bias3 = biases['out'].eval(sess)
        file_handle.write("The lenth of wight1 is {}\n".format(len(weight1)))
        file_handle.write("The lenth of wight2 is {}\n".format(len(weight2)))
        file_handle.write("The lenth of wight1 is {}\n".format(len(weight3)))
        # print("Training Finished........")
        # print("The lenth of wight1 is {}".format(len(weight1)))
        # print("The lenth of wight2 is {}".format(len(weight2)))
        # print("The lenth of wight1 is {}".format(len(weight3)))
        # print("Writing out parameters to w_20200106_h1_512_ep20000")
        # CurrentDateString = "{}".format(str(date.today()).replace("-", "")) + "_{}".format(str(time.time()).replace('.',''))
        CurrentDateString = "{}_{}".format(str(date.today()).replace("-", ""),datetime.now().strftime("%H_%M_%S"))
        newdirpath = "TrainedModels/{}x{}_{}".format(h1_layer_num,h2_layer_num,CurrentDateString)
        os.mkdir("../"+newdirpath)
        spio.savemat("../weight/8+2band(25ms)/weight_{}x{}_{}.mat".format( h1_layer_num, h2_layer_num, CurrentDateString),
                     {'w1': weight1, 'w2': weight2, 'w3': weight3, 'b1': bias1, 'b2': bias2, 'b3': bias3})

        saver.save(sess, "../{}/model_{}x{}_{}".format(newdirpath, h1_layer_num, h2_layer_num, CurrentDateString))
        file_handle.write("*********** End Of Training Of Learning Rate {} **********\n".format(current_lr_str))
        try:
            plt.plot(range(len(error_train)), error_train, 'b', label='Training accuracy')
            plt.title('accuracy')
            plt.xlabel('epoch', fontsize=16)
            plt.ylabel('accuracy', fontsize=16)
            plt.legend()
            plt.figure()
            plt.savefig("../expImg/exp_{}x{}_{}.png".format(h1_layer_num, h2_layer_num, CurrentDateString))
        except:
            pass
        finally:
            return

        # a = runMeanFast(error_train, 100)
        # b = runMeanFast(error_test, 100)
        # plt.plot(range(len(a)), a, 'b', label="train")
        # plt.grid()
        # plt.xlabel('epoch', fontsize=16)
        # plt.ylabel('accuracy', fontsize=16)
        # plt.xlim(0, 15900)
        # plt.xlim(0,4900)
        # plt.legend()
        # plt.figure()
        # plt.show()

ParameterList = [

    {
        "LearningRate":"0.008",
        "h1_layer":512,
        "h2_layer":32,
        "batch_size":40,
         "epoches":100
    },
    {
        "LearningRate":"0.008",
        "h1_layer":256,
        "h2_layer":64,
        "batch_size":40,
         "epoches":100
    }
]


# time_log_file = "../time_log/timelog_{}.log".format(str(date.today()).replace("-", ""))

time_log_file = "../time_log/timelog_{}_{}.log".format("{}_{}".format(str(date.today()).replace("-", ""),time.time()),str(time.time()).replace(".",""))
if __name__ == "__main__":
    with open(time_log_file, "a+") as f:
        start = timeit.default_timer()
        for i in range(len(ParameterList)):
            current_dict = ParameterList[i]
            h1_layer_num = current_dict["h1_layer"]
            h2_layer_num = current_dict["h2_layer"]
            batch_size = current_dict["batch_size"]
            trainProcessEntry(h1_layer_num=h1_layer_num, h2_layer_num=h2_layer_num, \
                          batch_size=batch_size , file_handle=f)
            print("h1_layer_num:{}, h2_layer_num:{}".format(h1_layer_num,h2_layer_num))
            stop = timeit.default_timer()
            total_training_time = stop - start
            f.write("Total Training Time is {} seconds.\n".format(total_training_time))
            print("Training Time: {}".format(total_training_time))

