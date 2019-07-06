from os.path import join, isfile
import numpy as np
from itertools import combinations
import collections
import matplotlib.pyplot as plt

# Phase 1
# DATA_DIR = 'Data'
# ORIGPATH = join(DATA_DIR) + '/Orig'

# Phase 2
DATA_DIR = 'DATA'
DATAPATH = join(DATA_DIR) + '/PH1'


def hex2bin(hex_value):
    """
    Convert from hexadecimal to binary vector

    :param hex_value: Hexadecimal value to be converted to binary vector
    """
    dec_val = int(hex_value, 16)
    return(bin(dec_val)[2:].zfill(16))


def main():

    features = []

    if isfile(DATAPATH + '/' + 'dataset6'):
        with open(DATAPATH + '/' + 'dataset6', 'r') as f:
            for line in f.readlines():
                inter_binv = hex2bin(line.split(' ')[0][0:4])
                bin_vector = [int(x) for x in inter_binv]
                features.append(np.array(bin_vector))
    else:
        print('File not found')
        exit(1)

    features = np.array(features)
    print(features)
    npsum = []

    for (i, j) in combinations(list(range(len(features))), 2):
        npsum.append(np.dot(features[i], features[j]))

    print(max(npsum), min(npsum), sum(npsum) / len(npsum))

    # npsum = [j for j in sorted(npsum)]
    # counter = collections.Counter(npsum)
    # x = list(counter.keys())
    # x_tick_poss = [i - 0.2 for i in x]
    # x_tick_vals = [str(i) for i in x]
    # height = list(counter.values())
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # fig.subplots_adjust(top=0.85)
    # ax.set_title('npsum Histogram')
    # ax.set_ylabel('Frequency')
    # ax.set_xlabel('Npsum value')
    # ax.bar(x, height=height, tick_label=x_tick_vals)
    # # plt.xticks(x, x_tick_vals)
    # for a, b in zip(x_tick_poss, height):
    #     plt.text(a, b, str(b))

    # plt.show()


if __name__ == '__main__':
    main()
