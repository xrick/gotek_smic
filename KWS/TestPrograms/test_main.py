import numpy as np
import yaml
import scipy.io as spio
import scipy.io.wavfile as wavio

w1 = None
w2 = None
w3 = None
w4 = None
b1 = None
b2 = None
b3 = None
b4 = None

def nth_root(num,root):
   answer = num ** (1/root)
   return answer

def read_paths():
    pass

def read_weights():
    pass

def read_biases():
    pass

def get_mfcc(sig, sr, frame_size, hop_length):
    pass

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
    return np.maximum(0,X)

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum

def load_weights(weights_file = None):
    weights = spio.loadmat(weights_file)
    # w1,w2,w3,w4,b1,b2,b3,b4 =
    return weights['w1'],weights['w2'],weights['w3'],weights['w4'],weights['b1'],weights['b2'],weights['b3'],weights['b4']

def test_logic(signal):
    # weight_file_ = ""
    # w1, w2, w3, w4, b1, b2, b3, b4 = load_weights()
    count_pass_all = []
    count_fail_all = []
    if len(signal) > 16000:
        signal = signal[0:160000]

    fs = 16000
    threshold = 0.25
    windowSize = fs * 0.025 #400
    windowStep = fs * 0.010 #160
    nDims = 40
    context_l = 30 # past 30 frames
    context_r = 10 # 10 future frames
    w_smooth = 10
    w_max = 100
    n_class = 3
    target = 1
    # preprocessing

    rest_signal = signal
    mfcc_coff = get_mfcc(signal, fs, windowSize, windowSize)
    temp = mfcc_coff, np.amin(mfcc_coff)
    mfcc_coff = np.divide(temp, np.amax(temp))
    nframe = len(mfcc_coff)
    """
    given a wav length L = 16000 frames
    L = 16000
    l = 400 (frame length)
    s = 160 (hop length, stride)
    then we can calculate the total Frames T with overlapping
    T = ((L-l) / s)+ 1
    we get T = 98
    """
    """
      ***....**             ***..**  
      ***....**             ***..**
    [ .      ..  mfcc_coff  .    .. ]
      .      ..    40x98    .    ..
      ***....**             ***..**
        40x30                40x10
        
    """
    input_coff = np.insert(np.zeros(nDims, context_l), mfcc_coff)
    input_coff = np.insert(input_coff, np.zeros(nDims, context_r))
    x_data = []
    x = []
    for context in range(nframe):
        xx = []
        window = mfcc_coff[:,context : context+context_l+context_r]
        for w in range(context_l+context_r):
            np.insert(xx, window[:, w]) # get the w-th column
        x = np.insert(x, xx)

    x_data = np.insert(x_data,np.transpose(x)) # 一個音檔的data
    all_answer = []
    all_smooth_answer = []
    all_confi = []
    y_test = []
    x_data_len = len(x_data)

    for j in range(x_data[:,1]):
        pred = relu(relu(relu(x_data[j, :]*w1 + b1)*w2 + b2)*w3 + b3)*w4 + b4
        answer = softmax(np.transpose(pred))
        all_answer = np.insert(all_answer, answer)

        # smooth
        if j <= w_smooth:
            smooth_answer = (1 / (j - 1 + 1)) * sum(all_answer[:, 1: j], 2)
            all_smooth_answer = [all_smooth_answer, smooth_answer]
        else:
            k = j - w_smooth + 1
            smooth_answer = (1 / (j - k + 1)) * sum(all_answer[:, k: j], 2)
            all_smooth_answer = [all_smooth_answer, smooth_answer]

        # calculate confidence, please reference the paper "small footprint kws using deep neural networks
        if j <= w_max:
            confi = nth_root(np.amax(all_smooth_answer[1, 1:j])*np.amax(all_smooth_answer[2, 1: j]), n_class - 1)
            all_confi = [all_confi, confi]
        else:
            k = j - w_max + 1
            confi = nth_root(np.amax(all_smooth_answer[1, k:j])*np.amax(all_smooth_answer[2, k: j]), n_class - 1)
            all_confi = np.insert(all_confi, confi)

    if  answer[0] > answer[1] and answer[0] > answer[2]:
        y_test = [y_test, 0]
    elif answer[1] > answer[0] and answer[1] > answer[2]:
        y_test = [y_test, 1]
    else:
        y_test = [y_test, 2]

    count_pass = 0
    count_fail = 0
    for l in range(len(all_confi)):
        if all_confi[l] > threshold:
            count_pass = count_pass + 1
            break

    if count_pass == 0:
        count_fail = count_fail + 1

    count_pass_all = np.insert(count_pass_all, count_pass)
    count_fail_all = np.insert(count_fail_all, count_fail)

    c_pass = sum(count_pass_all)
    c_fail = sum(count_fail_all)
    if target == 1:
        acc = sum(count_pass_all) / len(count_pass_all)
    else:
        acc = sum(count_fail_all) / len(count_fail_all)




def run_test_main():
    target = 1
    test_data_path = "path_to_data_path"
    # load test data into list
    fnlist = read_paths()
    w1, w2, w3, w4, b1, b2, b3, b4 = read_weights()
    # biases = read_biases()
    for f in fnlist:
        sig, sr = wavio.read(f)
        test_logic(sig)

if __name__ == "__main__":
    run_test_main()