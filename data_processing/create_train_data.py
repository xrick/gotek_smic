import numpy as np
import scipy.io.wavfile as wavio
import scipy.io as spio
import os
import sys
script_path = "../"
script_path2 = "../Libs/"
sys.path.append(os.path.abspath(script_path))
sys.path.append(os.path.abspath(script_path2))
from scipy import signal
from numpy.linalg import norm
from utils import get_recursive_files
import librosa
# import yaml

fs = 16000
windowSize = fs * 0.025
windowStep = fs * 0.010
nDims = 40
context_l = 30
context_r = 10
keyword_path = "path to keyword"
filler_path = "path to filler"
silence_path = "path to silence train data"
x_class1 = []
y_class1 = []

def resample_by_interpolation(signal, input_fs, output_fs):

    scale = output_fs / input_fs
    # calculate new length of sample
    n = round(len(signal) * scale)

    # use linear interpolation
    # endpoint keyword means than linspace doesn't go all the way to 1.0
    # If it did, there are some off-by-one errors
    # e.g. scale=2.0, [1,2,3] should go to [1,1.5,2,2.5,3,3]
    # but with endpoint=True, we get [1,1.4,1.8,2.2,2.6,3]
    # Both are OK, but since resampling will often involve
    # exact ratios (i.e. for 44100 to 22050 or vice versa)
    # using endpoint=False gets less noise in the resampled sound
    resampled_signal = np.interp(
        np.linspace(0.0, 1.0, n, endpoint=False),  # where to interpret
        np.linspace(0.0, 1.0, len(signal), endpoint=False),  # known positions
        signal,  # known data points
    )
    return resampled_signal

def vad_test(s, fs):
    s = s - np.amin(s)
    s = s / np.amax(s)
    FrameSize = int(fs * 0.025)  # 400
    ShiftSize = int(fs * 0.010)  # 160
    Overlap = FrameSize - ShiftSize  # 240
    threshold = -1.9
    s_temp = []
    temp = []
    temp_all = []
    new = []
    rest_s = []
    t = s
    n = np.floor((len(s) - FrameSize) / ShiftSize)  # 97
    loop_size = int(ShiftSize * n + FrameSize)  # 15920
    norm_t = norm(t, 2)  # 115.2325447
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

    end = len(new)  # len(s_temp)
    s_temp = s_temp[0:end]  # s_temp[0:(end - Overlap)]
    new_s = np.transpose(new) * s_temp

    for j in range(len(new)):
        if new[j] == 1:
            rest_s = np.hstack((rest_s, new_s[j]))
            # rest_s = np.insert(rest_s, new_s[j])
    return rest_s

def get_mel_fb(SR=16000, num_fft=1204, num_mels=13, F_Min=133, F_Max=933):


def get_mfcc_librosa(wav_sig=None, sample_rate=16000, frame_length=400, step_length=160,
                     num_mels=40, num_mfccs=40, mel_fb=None, dct_type=2, window='hamming'):
    tmp_melspec = librosa.feature.melspectrogram(y=wav_sig, sr=sample_rate,
                                                 S=mel_fb,
                                                 n_mels=num_mels,
                                                 n_fft=1024,
                                                 hop_length=step_length,
                                                 win_length=frame_length,
                                                 window=window)

    tmp_melspec = librosa.power_to_db(tmp_melspec)
    _mfcc = librosa.feature.mfcc(S=tmp_melspec, dct_type=2, n_mfcc=num_mfccs, norm=None, lifter=0)
    return _mfcc

def main_entry():
    keyword_files = get_recursive_files(keyword_path)
    filler_files = get_recursive_files(filler_path)
    silence_files = get_recursive_files(silence_path)
    for k in keyword_files:
        sig_, sr_ = wavio.read(k)
        if sr_ != 16000:
            sig_ = resample_by_interpolation(sig_, sr_, 16000)
        removed_sig = vad_test(sig_, 16000)
        len_of_sig_ = len(removed_sig)
        if len_of_sig_ > 8000:
            removed_sig = removed_sig[(len_of_sig_ - 8000):len_of_sig_, :]
        elif len_of_sig_ < 8000:
            removed_sig = np.hstack(0.5 + np.random.rand(8000 - len_of_sig_, 1) * 10 ^ -6, removed_sig)

        coeff = get_mfcc_librosa(wav_sig=removed_sig, window=None)
        temp = coeff(1,:)-min(coeff(1,:));
        coeff(1,:) = temp. / max(temp);
        nframe = length(coeff(1,:));
        coeff = [zeros(nDims, context_l), coeff, zeros(nDims, context_r)];

        x = [];
        y = [];





if __name__ == "__main__":
    main_entry()