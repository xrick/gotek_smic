import numpy as np
import scipy.io as spio

def load_weights(weightFile):
    dict_data = spio.loadmat(weightFile)
    w1 = dict_data["w1"]
    w2 = dict_data["w2"]
    w3 = dict_data["w3"]
    w4 = dict_data["w4"]
    b1 = dict_data["b1"]
    b2 = dict_data["b2"]
    b3 = dict_data["b3"]
    b4 = dict_data["b4"]
    return w1, w2, w3, w4, b1, b2, b3, b4

def quant_weight(ori_weight, bits, typeStr):
    min_wt = np.amin(ori_weight)
    max_wt = np.amax(ori_weight)
    # calculat the bits necessary for representing the range
    repr_bits = int(np.ceil(np.log2(max(abs(min_wt),abs(max_wt)))))
    frac_bits = bits-repr_bits-1
    quant_weight = np.round(ori_weight*(2**frac_bits)).astype(typeStr)
    # we need to return the frac_bits for the reason that when we do the inference
    # we need to scale the quantized weights to their original range by
    # weight = quant_weight * 2^frac_bits
    return quant_weight, frac_bits


def performConversion(srcWeightFile, bits, typeStr, des_file):
    w1, w2, w3, w4, b1, b2, b3, b4 = load_weights(srcWeightFile)
    quant_w1, frac_bits1 = quant_weight(w1, bits, typeStr)
    quant_w2, frac_bits2 = quant_weight(w2, bits, typeStr)
    quant_w3, frac_bits3 = quant_weight(w3, bits, typeStr)
    quant_w4, frac_bits4 = quant_weight(w4, bits, typeStr)

    quant_b1, frac_bits5 = quant_weight(b1, bits, typeStr)
    quant_b2, frac_bits6 = quant_weight(b2, bits, typeStr)
    quant_b3, frac_bits7 = quant_weight(b3, bits, typeStr)
    quant_b4, frac_bits8 = quant_weight(b4, bits, typeStr)
    print("\nquant_w2 is\n{}".format(quant_w2))
    print("\nquant_scale for w1 is\n{}".format(frac_bits1))
    print("\nquant_scale for w2 is\n{}".format(frac_bits2))
    print("\nquant_scale for w3 is\n{}".format(frac_bits3))
    print("\nquant_scale for w4 is\n{}".format(frac_bits4))
    print("\nquant_scale for b1 is\n{}".format(frac_bits5))
    print("\nquant_scale for b2 is\n{}".format(frac_bits6))
    print("\nquant_scale for b3 is\n{}".format(frac_bits7))
    print("\nquant_scale for b4 is\n{}".format(frac_bits8))
    spio.savemat(des_file, {
        'w1': quant_w1, 'w2': quant_w2, 'w3': quant_w3, 'w4': quant_w4,
        'b1': quant_b1, 'b2': quant_b2, 'b3': quant_b3, 'b4': quant_b4,
        's1': frac_bits1, 's2': frac_bits2, 's3': frac_bits3, 's4': frac_bits4,
        's5': frac_bits5, 's6': frac_bits6, 's7': frac_bits7, 's8': frac_bits8
    })

def write_out_c_weights(srcWeightMatFile):
    qw1,qw2,qw3,qw4,qb1,qb2,qb3,qb4=load_weights(srcWeightMatFile)
    # with open("../kws_c_weights_{}")