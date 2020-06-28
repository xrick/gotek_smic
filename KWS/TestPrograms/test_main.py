import numpy as np
import yaml
import scipy.io as spio
import scipy.io.wavfile as wavio

def read_paths():
    pass

def read_weights():
    pass

def read_biases():
    pass

def get_mfcc(sig, sr, frame_size, hop_length):
    pass

def test_logic(signal, smplerate):
    if len(signal) > 16000:
        signal = signal[0:160000]

    fs = 16000
    windowSize = fs * 0.025 #400
    windowStep = fs * 0.010 #160
    nDims = 40
    context_l = 30 # past 30 frames
    context_r = 10 # 10 future frames

    # preprocessing
    x_data = []
    rest_signal = signal
    mfcc_coff = get_mfcc(signal, fs, windowSize, windowSize)
    temp = np.subtract(mfcc_coff - np.amin(mfcc_coff))
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

    x = [];
    for context in range(nframe):
        xx = []
        window = mfcc_coff[:,context : context+context_l+context_r]
        for w in range(context_l+context_r):
            np.insert(xx,window[:,w]) # get the w-th column
        x = np.insert(x,xx)
        # x = [x,xx]
    x_data = np.insert(x_data,np.transpose(x)) # 一個音檔的data



def run_test_main():
    target = 1
    test_data_patj = "path_to_data_path"
    # load test data into list
    fnlist = read_paths()
    weights = read_weights()
    biases = read_biases()
    for f in fnlist:
        sig, sr = wavio(f)
        test_logic(sig, sr)

if __name__ == "__main__":
    run_test_main()