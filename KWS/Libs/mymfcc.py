import numpy as np
from scipy.fftpack import dct
import os
import sys
scriptpath = "../"
sys.path.append(os.path.abspath(scriptpath))
import math
import decimal
import logging


def hz2mel_nature(freq):
    return 1127. * np.log(1. + freq / 700.)

def mel2hz_nature(mel):
    return 700. * (np.exp(mel / 1127.) - 1.)

def hz2mel(hz):
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    return 700*(10**(mel/2595.0)-1)

def round_half_up(number):
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))


def get_default_filterbanks(nfilt=10, nfft=1024, samplerate=16000, lowfreq=0, highfreq=8000):
    highfreq = highfreq or samplerate / 2
    assert highfreq <= samplerate / 2, "highfreq is greater than samplerate/2"
    # compute points evenly spaced in mels
    lowmel = hz2mel_nature(lowfreq)
    highmel = hz2mel_nature(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    mid_freqs = mel2hz_nature(melpoints)

    bins = np.floor((nfft + 1) * mid_freqs / samplerate)
    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(0, nfilt):
        for i in range(int(bins[j]), int(bins[j + 1])):
            fbank[j, i] = (i - bins[j]) / (bins[j + 1] - bins[j])
        for i in range(int(bins[j + 1]), int(bins[j + 2])):
            fbank[j, i] = (bins[j + 2] - i) / (bins[j + 2] - bins[j + 1])
    return fbank

def get_filterbank_from_midfreqs(midFreqs,samplerate, n_filt, n_fft):
#     mid_freqs = midFreqs#[229.8,304.1,402.4,532.4,704.4,931.9,1233.1,1631.5,4000.,5500.]
    target_mid_freqs = np.empty(n_filt+2,dtype=np.float)
    idx = 0
    for freq in midFreqs:
        target_mid_freqs[idx] = freq
        idx += 1
#     print(target_mid_freqs)
    bins = np.floor((n_fft+1)*target_mid_freqs/samplerate)
#     print(len(bins))
    fbank = np.zeros([n_filt,n_fft//2+1])
    for j in range(0,n_filt):
        for i in range(int(bins[j]), int(bins[j+1])):
            fbank[j,i] = (i - bins[j]) / (bins[j+1]-bins[j])
        for i in range(int(bins[j+1]), int(bins[j+2])):
            fbank[j,i] = (bins[j+2]-i) / (bins[j+2]-bins[j+1])
    return fbank

def magspec(frames, NFFT):
    if np.shape(frames)[1] > NFFT:
        logging.warn(
            'frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.',
            np.shape(frames)[1], NFFT)
    complex_spec = np.fft.fft(frames, NFFT)
    return np.absolute(complex_spec)

def powspec(frames, NFFT):
    theFrames = magspec(frames,NFFT)
    energy = np.sum(theFrames,1)
    # return theFrames, energy
    return theFrames**2, energy
#     return 1.0 / NFFT * numpy.square(theFrames)

def log10powspec(frames, NFFT=1024, norm=0):
    ps, _ = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = 10.0 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def logpowspec(frames, NFFT=1024, norm=0):
    ps, _ = powspec(frames, NFFT)
    ps[ps <= 1e-30] = 1e-30
    lps = np.log(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps

def max_norm_signal(frames):
    return frames/np.max(frames)

def mean_norm_signal(frames):
    mean_frames = frames - (np.mean(frames, axis=0) + 1e-8)
    return mean_frames

def min_max_norm_signal(frames):
    min_ele = np.amin(frames)
    max_ele = np.amax(frames)
    norm_frames = (frames - min_ele) / (max_ele - min_ele)
    return norm_frames

def FormatWavSig_MS(norm_sig, sr):
    strt_samp = 0
    end_samp = len(norm_sig)
    end_ms = len(norm_sig)/sr
    xrange = np.linspace(0, end_ms, end_samp-strt_samp)
    return strt_samp, end_samp, end_ms, xrange


def framesig(sig, frame_len, frame_step):
    slen = len(sig)
    #     frame_len = int(round_half_up(frame_len))
    #     frame_step = int(round_half_up(frame_step))
    if slen <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(math.ceil((1.0 * slen - frame_len) / frame_step))

    padlen = int((numframes - 1) * frame_step + frame_len)

    zeros = np.zeros((padlen - slen,))
    padsignal = np.concatenate((sig, zeros))

    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(
        np.arange(0, numframes * frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    return frames

def get_mfcc(spectrum=None,dct_type=2,num_ceps=13):
    ret_mfcc = dct(spectrum, type=dct_type, axis=1, norm="ortho")[:, 1 : (num_ceps + 1)]
    return ret_mfcc

def get_min_max_norm_mfcc(spectrum,dct_type=2,num_ceps=13):
    _mfcc = dct(spectrum, type=dct_type, axis=1, norm=None)[:, 1 : (num_ceps + 1)]
    min_ele = np.amin(_mfcc)
    max_ele = np.amax(_mfcc)
    norm_mfcc = (_mfcc-min_ele)/(max_ele-min_ele)
    return norm_mfcc

def get_max_norm_mfcc(spectrum, dct_type=2, num_ceps=13):
    _mfcc = dct(spectrum, type=dct_type, axis=1, norm=None)[:, 1 : (num_ceps + 1)]
    max_ele = np.amax(_mfcc)
    norm_mfcc = _mfcc/max_ele
    return norm_mfcc

def get_mean_mfcc(mfccs):
    mean_mfccs = mfccs - (np.mean(mfccs, axis=0) + 1e-8)
    return mean_mfccs

def lifter(cepstra, L=22):
    """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
    magnitude of the high frequency DCT coeffs.
    :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
    :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    """
    if L > 0:
        nframes,ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L/2.)*np.sin(np.pi*n/L)
        return lift*cepstra
    else:
        # values of L <= 0, do nothing
        return cepstra

def delta(feat, N):
    """Compute delta features from a feature vector sequence.
    :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
    :param N: For each frame, calculate delta features based on preceding and following N frames
    :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row del sig
    del samp_freq
    del sig_len
    del frame_len
    del over_lap
    del step_len
    del framed_sig
    del frame_lps, _energy
    del half_frame_lps
    del wav_feat
    del norm_wav_mfcc
    print(flatten_mfcc.shape)
    return flatten_mfccholds 1 delta feature vector.
    """
    if N < 1:
        raise ValueError('N must be an integer >= 1')
    NUMFRAMES = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = np.empty_like(feat)
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
    for t in range(NUMFRAMES):
        # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator
    return delta_feat