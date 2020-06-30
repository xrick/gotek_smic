import numpy as np
from numpy.linalg import norm
import scipy.io.wavfile as wavio

def vad_test(s, fs):
    s = s - np.amin(s)
    s = s / np.amax(s)
    FrameSize = int(fs * 0.025) #400
    ShiftSize = int(fs * 0.010) #160
    Overlap = FrameSize - ShiftSize #240
    threshold = -1.9
    s_temp = []
    temp = []
    temp_all = []
    new = []
    rest_s = []
    t = s
    n = np.floor((len(s) - FrameSize) / ShiftSize) #97
    loop_size = int(ShiftSize * n + FrameSize) #15920
    norm_t = norm(t, 2) #115.2325447
    for i in range(FrameSize, loop_size, ShiftSize):
        temp = np.log(norm(t[i - FrameSize:i], 2) / norm_t + 0.00001)
        # temp_all = np.insert(temp_all, temp)#[temp_all, temp]
        temp_all = np.hstack((temp_all, temp))
        if temp > threshold:
            # new = [new, 1 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 1 * np.ones(ShiftSize)))
        else:
            # new = [new, 0 * np.ones(ShiftSize, 1)]
            new = np.hstack((new, 0 * np.ones(ShiftSize)))

    # for i in range(ShiftSize * n + FrameSize):
    # for i in range(loop_size): #15920
    s_temp = np.array(s)

    end = len(new)#len(s_temp)
    s_temp = s_temp[0:end ]#s_temp[0:(end - Overlap)]
    new_s = np.transpose(new) * s_temp

    for j in range(len(new)):
        if new[j] == 1:
            rest_s = np.hstack((rest_s, new_s[j]))
            # rest_s = np.insert(rest_s, new_s[j])

    return rest_s

if __name__ == "__main__":
    # read test file
    test_wav = "../../Speech_DataSets/whole_keyword_clean_second_run_1429/0b40aa8e_nohash_0y4s6_1.wav"
    save_wav = "../../Speech_DataSets/whole_keyword_clean_second_run_1429/silence_removed/reduced_0b40aa8e_nohash_0y4s6_1.wav"
    fs, sig = wavio.read(test_wav)
    processed_sig = vad_test(sig, fs)
    print("original signal length is {}".format(sig.shape))
    print("processed signal length is {}".format(processed_sig.shape))
