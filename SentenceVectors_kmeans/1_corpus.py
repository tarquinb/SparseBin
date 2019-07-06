import os
import numpy as np
import pandas as pd

DATAPATH = os.path.join('DATA')
VOCAB = [chr(i) for i in range(ord('a'), ord('q'))]


def hex2bin(hex_value):
    """
    Convert from hexadecimal to binary vector

    :param hex_value: Hexadecimal value to be converted to binary vector
    """
    dec_val = int(hex_value, 16)
    return(bin(dec_val)[2:].zfill(16))


def datalist(filename):
    DATAFILE = os.path.join('DATA') + '/' + filename
    datalist = []
    with open(DATAFILE, 'r') as df:
        for line in df.readlines():
            datalist.append(line)
    return datalist


def data(filename, data_pos):
    DATAFILE = DATAPATH + '/' + filename
    features = []
    labels = []
    with open(DATAFILE, 'r') as df:
        for line in df.readlines():
            labels.append(int(line.split(' ')[1]))
            if data_pos == 0:
                inter_binv = hex2bin(line.split(' ')[0][0:4])
            elif data_pos == 1:
                inter_binv = hex2bin(line.split(' ')[0][2:6])
            elif data_pos == 2:
                inter_binv = hex2bin(line.split(' ')[0][4:8])
            bin_vector = [int(x) for x in inter_binv]
            bin_vector = np.array(bin_vector)
            features.append(bin_vector)

    labels = np.vstack(np.array(labels))
    features = np.vstack(np.array(features))
    return features, labels


def get_corpus(filename, data_pos):
    features, labels = data(filename, data_pos)
    index = range(1, len(features) + 1)
    columns = ['sentence', 'label']
    df = pd.DataFrame(index=index, columns=columns)
    for i in range(np.shape(features)[0]):
        sentence = []
        df.loc[i + 1]['label'] = labels[i][0]
        for n in range(len(features[i])):
            if features[i][n] == 1:
                sentence.append(VOCAB[n])
            else:
                continue
        df.loc[i + 1]['sentence'] = sentence

    return df
