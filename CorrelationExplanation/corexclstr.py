import corex as ce
import numpy as np
import os


def hex2bin(hex_value):
    """
    Convert from hexadecimal to binary vector

    :param hex_value: Hexadecimal value to be converted to binary vector
    """
    dec_val = int(hex_value, 16)
    return(bin(dec_val)[2:].zfill(16))


def data(filename):
    DATAFILE = os.path.join('DATA') + '/' + filename
    features = []
    datalist = []
    with open(DATAFILE, 'r') as df:
        for line in df.readlines():
            datalist.append(line)
            inter_binv = hex2bin(line.split(' ')[0][4:8])
            bin_vector = [int(x) for x in inter_binv]
            bin_vector = np.array(bin_vector)
            features.append(bin_vector)

    features = np.vstack(np.array(features))
    return features, datalist


def main():
    features, datalist = data('jeu_hota4')

    layer1 = ce.Corex(n_hidden=2, dim_hidden=4, verbose=1, seed=123)
    layer1.fit(features)

    for i in range(len(datalist)):
        if layer1.labels[i][0] == 0:
            with open('JH4_CL0', 'a') as f:
                f.write(datalist[i].strip('\n') + '\n')
        elif layer1.labels[i][0] == 1:
            with open('JH4_CL1', 'a') as f:
                f.write(datalist[i].strip('\n') + '\n')
        elif layer1.labels[i][0] == 2:
            with open('JH4_CL2', 'a') as f:
                f.write(datalist[i].strip('\n') + '\n')
        elif layer1.labels[i][0] == 3:
            with open('JH4_CL3', 'a') as f:
                f.write(datalist[i].strip('\n') + '\n')


if __name__ == '__main__':
    main()
