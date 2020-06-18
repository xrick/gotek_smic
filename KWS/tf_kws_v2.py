# import tensorflow as tf
import tensorflow.compat.v1 as tf
import numpy as np
import os
import sys
import json
import scipy.io as spio

# class DNN128(object):
#     def __init__(self, model_settings=None):
#         self.NN_Name = "DNN128_3"
#         self.setting = None
#         self.setting_file = None
#         if model_settings is not None:
#             self.setting_file = model_settings
#             self.setting = read_nn_config(self.setting_file,self.NN_Name)
#         self.lr = 0.001
#         self.loss = None
#         self.optimizer = 'gradientdescent'
#         self.n_band = 40*41
#         self.n_h1 = 128
#         self.n_h2 = 128
#         self.n_h3 = 128
#         self.n_class = 3
        
#     def get_init_weights(self):
        
#         weights = {
#             'hidden1': tf.Variable(tf.random_normal([self.n_band, self.n_h1],dtype=tf.float32,stddev=0.1)),
#             'hidden2': tf.Variable(tf.random_normal([self.n_h1, self.n_h2],dtype=tf.float32,stddev=0.1)),
#             'hidden3': tf.Variable(tf.random_normal([self.n_h2, self.n_h3],dtype=tf.float32,stddev=0.1)),
#             'output': tf.Variable(tf.random_normal([self.n_h3, self.n_class],dtype=tf.float32,stddev=0.1))
#         }
#         biases = {
#             'hidden1': tf.Variable(tf.zeros([n_h1],dtype=tf.float32)),
#             'hidden2': tf.Variable(tf.zeros([n_h2],dtype=tf.float32)),
#             'hidden3': tf.Variable(tf.zeros([n_h3],dtype=tf.float32)),
#             'output': tf.Variable(tf.zeros([n_class],dtype=tf.float32))
#         }
#         return weights, biases

#     def init_weights_with_pretrained_weights(self,pretrained_weights_list):
#         pass


#     def get_dnn_model_v2(fingerprint_input, model_settings, model_size_info, 
#                        is_training):
#         """Builds a model with multiple hidden fully-connected layers.
#         model_size_info: length of the array defines the number of hidden-layers and
#                         each element in the array represent the number of neurons 
#                         in that layer 
#         """
#         if is_training:
#             dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
#         fingerprint_size = model_settings['fingerprint_size']
#         label_count = model_settings['label_count']
#         num_layers = len(model_size_info)
#         layer_dim = [fingerprint_size]
#         layer_dim.extend(model_size_info)
#         flow = fingerprint_input
#         tf.summary.histogram('input', flow)
#         for i in range(1, num_layers + 1):
#             with tf.variable_scope('fc'+str(i)):
#                 W = tf.get_variable('W', shape=[layer_dim[i-1], layer_dim[i]], 
#                         initializer=tf.contrib.layers.xavier_initializer())
#                 tf.summary.histogram('fc_'+str(i)+'_w', W)
#                 b = tf.get_variable('b', shape=[layer_dim[i]])
#                 tf.summary.histogram('fc_'+str(i)+'_b', b)
#                 flow = tf.matmul(flow, W) + b
#                 flow = tf.nn.relu(flow)
#                 if is_training:
#                     flow = tf.nn.dropout(flow, dropout_prob)
#         weights = tf.get_variable('final_fc', shape=[layer_dim[-1], label_count], 
#                     initializer=tf.contrib.layers.xavier_initializer())
#         bias = tf.Variable(tf.zeros([label_count]))
#         logits = tf.matmul(flow, weights) + bias
#         if is_training:
#             return logits, dropout_prob
#         else:
#             return logits

#     def get_dnn_model(self):
#         w, b = self.get_init_weights()
#         x = tf.placeholder(tf.float32, [None, self.n_band])
#         y = tf.placeholder(tf.float32, [None, self.n_class])
#         x1 = tf.nn.relu(tf.add(tf.matmul(x,w['hidden1']),b['hidden1']))
#         x2 = tf.nn.relu(tf.add(tf.matmul(x1,w['hidden2']),b['hidden2']))
#         x3 = tf.nn.relu(tf.add(tf.matmul(x2,w['hidden3']),b['hidden3']))
#         logits = tf.add(tf.matmul(x3,w['output']),b['output'])
#         return logits 


tf.disable_v2_behavior()
tf.reset_default_graph()
"""
Define global variables
"""
checkpoint_directory = "../training_ckpt"
checkpoint_prefix = os.path.join(checkpoint_directory, "ckpt")

# Parameters
learning_rate = 0.001
n_band = 2240#40*56 # mfcc dim * context size #40*41
batch_size = 1280
display_step = 100
n_hidden1 = 128
n_hidden2 = 128
n_hidden3 = 128
n_classes = 3


def read_nn_config(config,nn_name):
    data = None
    with open(config) as f:
        data = json.load(f)
    for d in data:
        if d["NN_Name"] == "DNN128_3":
            return d
        return None


def get_init_weights():
        
        weights = {
            'h1': tf.Variable(tf.compat.v1.random_normal([n_band, n_hidden1],dtype=tf.float32,stddev=0.1)),
            'h2': tf.Variable(tf.compat.v1.random_normal([n_hidden1, n_hidden2],dtype=tf.float32,stddev=0.1)),
            'h3': tf.Variable(tf.compat.v1.random_normal([n_hidden2, n_hidden3],dtype=tf.float32,stddev=0.1)),
            'output': tf.Variable(tf.compat.v1.random_normal([n_hidden3, n_classes],dtype=tf.float32,stddev=0.1))
        }
        biases = {
            'b1': tf.Variable(tf.zeros([n_hidden1],dtype=tf.float32)),
            'b2': tf.Variable(tf.zeros([n_hidden2],dtype=tf.float32)),
            'b3': tf.Variable(tf.zeros([n_hidden3],dtype=tf.float32)),
            'b_output': tf.Variable(tf.zeros([n_classes],dtype=tf.float32))
        }
        return weights, biases

def create_neural_net(x):
    """
    create a 128x128x128 DNN
    params:
    x : place_holder for x_train
    """
    weights, biases = get_init_weights()
    # Hidden fully connected layer with 128 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 128 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Hidden fully connected layer with 128 neurons
    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_3, weights['output']) + biases['output']
    return out_layer

def next_batch(num, data, labels):
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)


def exam_train_data(x, y):
    # print(x)
    f_pass = np.isnan(x.any())
    if f_pass:
        print(y)
        return True
    else:
        return False

def peek_train_data(epoach, step,x_n, y_n):
    print("epoach")

def App_Run():
    # def load_variables_from_checkpoint(sess, start_checkpoint):
    #     """Utility function to centralize checkpoint restoration.
    #     Args:
    #         sess: TensorFlow session.
    #         start_checkpoint: Path to saved checkpoint on disk.
    #     """
    #     saver = tf.train.Saver(tf.global_variables())
    #     saver.restore(sess, start_checkpoint)
    # train_log_f = open("./training_log/t_log_{}")
    data_file = "../train_data/20200605/shuffled_train_data.npy"
    lbl_file = "../train_data/20200605/shuffled_train_label.npy"
    val_data_file = "../train_data/20200605/shuffled_val_train_data.npy"
    val_lbl_file = "../train_data/20200605/shuffled_val_train_label.npy"
    x_train = np.load(data_file,allow_pickle=True)
    y_train = np.load(lbl_file,allow_pickle=True)
    # y_train = tf.keras.utils.to_categorical(y_train)
    # x_test = np.load(val_data_file,allow_pickle=True)
    # y_test = np.load(val_lbl_file,allow_pickle=True)
    # y_test = tf.keras.utils.to_categorical(y_test)
    X = tf.placeholder(tf.float32, [None, n_band])
    Y = tf.placeholder(tf.int8, [None, n_classes])
    logits = create_neural_net(X)
    prediction = tf.nn.softmax(logits)
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
    
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)
    correct_prediction = tf.equal(tf.argmax(prediction,1), tf.argmax(Y,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

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
    acc_now=0
    epochnum=3
    ckpt_file_path = "../training_ckpt/weights_improvement_{}-{}.ckpt"
    '''Add ops to save and restore all the variables.'''
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        
        '''    Restore variables from disk.    '''
    #    checkpoint = tf.train.latest_checkpoint("checkpoints/checkpoints_1017_256")
    #    saver.restore(sess, checkpoint)

        # ckpt_save_point = 100
        # run_count = 0
        current_epoch = 0
        for epoch in range(epochnum):
            current_epoch = epoch
            for step in range(100):
                batch_x, batch_y = next_batch(batch_size,x_train,y_train)
                peek_train_data(batch_x, batch_y)
                sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
                # valid_flag = exam_train_data(x_n, y_n)
                # if valid_flag:
                #     sess.run(train_op, feed_dict={x: x_n, y: y_n})
                #     # run_count += 1
                # else: continue
                if step % display_step == 0 or step == 1:
                    # Calculate batch loss and accuracy
                    loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                    estimated_pred = sess.run(prediction, feed_dict={x: x_train, y: y_train})
            # acc1 = sess.run(accuracy, feed_dict={x: x_test, y: y_test})

            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

            print("epoch",epoch)
            print("train : ",acc)
            print("test : ",acc1)
            
            error_train.append(acc)
            error_test.append(acc1)
            
            if acc>acc_now:
                acc_now=acc
                weight1=w['hidden1'].eval(sess)
                weight2=w['hidden2'].eval(sess)
                weight3=w['hidden3'].eval(sess)
                weight4=w['output'].eval(sess)
                bias1=b['hidden1'].eval(sess)
                bias2=b['hidden2'].eval(sess)
                bias3=b['hidden3'].eval(sess)
                bias4=b['output'].eval(sess)
                spio.savemat('kws_weights/w_3layer128.mat', {'w1': weight1,'w2':weight2,'w3':weight3,'w4':weight4,
                                                        'b1':bias1,'b2':bias2,'b3':bias3,'b4':bias4})
                saver.save(sess, ckpt_file_path.format(current_epoch,acc_now))
            
        
        # '''    # Save the variables to disk. '''
        #save_path = saver.save(sess, "checkpoints/checkpoints_1211/model_1211d_ep_{}.ckpt".format(epoch))
if __name__ == "__main__":
    App_Run()