import tensorflow as tf
from Models.BaseModel import BaseModel


class DNN128(BaseModel):
    def __init__(self, optimizer='gradientdescent', loss='crossentropy', model_settings=None, train=True,
                 num_hidden=128, num_layers=3):
        super(DNN128,self).__init__(optimizer, loss, model_settings, train, num_hidden, num_layers)
        self.n_band = model_settings['fingerprint_size']#3920 #40x98
        self.n_h1 = 128
        self.n_h2 = 128
        self.n_h3 = 128
        self.weights = None
        self.biases = None#self.get_init_weights(lable_len)

    def get_in_ground_truth(self):
        # X
        fingerprint_input = tf.placeholder(tf.float32, [None, self.model_settings['fingerprint_size']],
                                           name='fingerprint_input_' + self.model)
        # Y
        ground_truth_input = tf.placeholder(tf.int64, [None], name='ground_truth_input')
        return fingerprint_input, ground_truth_input, None
    def get_init_weights(self,label_len):
        weights = {
                    'h1': tf.Variable(tf.truncated_normal([self.n_band, self.n_h1],dtype=tf.float32,stddev=0.001)),
                    'h2': tf.Variable(tf.truncated_normal([self.n_h1, self.n_h2],dtype=tf.float32,stddev=0.001)),
                    'h3': tf.Variable(tf.truncated_normal([self.n_h2, self.n_h3],dtype=tf.float32,stddev=0.001)),
                    'output': tf.Variable(tf.truncated_normal([self.n_h3, label_len],dtype=tf.float32,stddev=0.001))
                }
        biases = {
                    'h1': tf.Variable(tf.zeros([self.n_h1],dtype=tf.float32)),
                    'h2': tf.Variable(tf.zeros([self.n_h2],dtype=tf.float32)),
                    'h3': tf.Variable(tf.zeros([self.n_h3],dtype=tf.float32)),
                    'output': tf.Variable(tf.zeros([label_len],dtype=tf.float32))
                }
        return weights, biases
    def get_logits_dropout(self, fingerprint_input, seq_len):
        if self.train:
            dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')
        # hidden_layer_len = 128
        # fingerprint_size = self.model_settings['fingerprint_size']
        label_count = self.model_settings['label_count']
        # weights = tf.Variable(tf.truncated_normal([fingerprint_size, label_count], stddev=0.001))
        self.weights, self.biases = self.get_init_weights(label_count)
        # bias = tf.Variable(tf.zeros([label_count]))
        # logits = tf.matmul(fingerprint_input, weights) + bias
        layer_1 = tf.add(tf.matmul(fingerprint_input, self.weights['h1']), self.biases['h1'])
        # Hidden fully connected layer with 128 neurons
        layer_2 = tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['h2'])
        # Hidden fully connected layer with 128 neurons
        layer_3 = tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['h3'])
        # Output fully connected layer with a neuron for each class
        logits = tf.matmul(layer_3, self.weights['output']) + self.biases['output']
        if self.train:
            return logits, dropout_prob
        else:
            return logits

    def get_confusion_matrix_correct_labels(self, ground_truth_input, logits, seq_len, audio_processor):
        predicted_indices = tf.argmax(logits, 1)
        correct_prediction = tf.equal(predicted_indices, ground_truth_input)
        confusion_matrix = tf.confusion_matrix(ground_truth_input, predicted_indices,
                                                        num_classes=self.model_settings['label_count'])
        return predicted_indices, correct_prediction, confusion_matrix

    # def save_weights(self, sess, weights_path):
    #     weight1 = self.weights['h1'].eval(sess)
    #     weight2 = self.weights['h2'].eval(sess)
    #     weight3 = self.weights['h3'].eval(sess)
    #     weight4 = self.weights['output'].eval(sess)
    #     bias1 = self.biases['h1'].eval(sess)
    #     bias2 = self.biases['h2'].eval(sess)
    #     bias3 = self.biases['h3'].eval(sess)
    #     bias4 = self.biases['output'].eval(sess)
    #     CurrentDateString = "{}_{}".format(str(date.today()).replace("-", ""),datetime.now().strftime("%H_%M_%S"))
    #     spio.savemat(weights_path.format(self.n_h1, self.n_h2, self.n_h3, CurrentDateString),
    #                  {'w1': weight1, 'w2': weight2, 'w3': weight3, 'w4': weight4, 'b1': bias1,
    #                   'b2': bias2, 'b3': bias3, 'b4': bias4})