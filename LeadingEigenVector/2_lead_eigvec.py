from os.path import join, isfile
from os import listdir
import numpy as np
from itertools import combinations
import networkx as nx
import _pickle as pickle
from numpy.linalg import eig

DATAPATH = join('DATA/')
DATA_PH1 = DATAPATH + 'PH1/'
GRPH_PH1 = DATAPATH + 'GRAPH/PH1/'
ADJM_PH1 = DATAPATH + 'ADJMAT/PH1/'
MODM_PH1 = DATAPATH + 'MODMAT/PH1/'
EIGN_PH1 = DATAPATH + 'EIGEN/PH1/'
DATA_PH2 = DATAPATH + 'PH2/'
GRPH_PH2 = DATAPATH + 'GRAPH/PH2/'
ADJM_PH2 = DATAPATH + 'ADJMAT/PH2/'
MODM_PH2 = DATAPATH + 'MODMAT/PH2/'
EIGN_PH2 = DATAPATH + 'EIGEN/PH2/'
RSLT_PH2 = join('RESULTS/PH2/')


def hex2bin(hex_value):
    """
    Convert from hexadecimal to binary vector

    :param hex_value: Hexadecimal value to be converted to binary vector
    """
    dec_val = int(hex_value, 16)
    return(bin(dec_val)[2:].zfill(16))


def preprocess(filepath):
    '''
    Convert the feature vectors from hexadecimal to 16-bit binary
    representations.

    :param filepath: Complete path of file containing dataset to be processed
    :return: Numpy array containing binary vectors and dictionary with original
             hexadecimal vectors as keys and labels as values
    '''

    features = []
    orig_dict = dict()
    if isfile(filepath):
        with open(filepath, 'r') as f:
            for line in f.readlines():
                inter_binv = hex2bin(line.split(' ')[0])
                orig_dict[line.split(' ')[0]] = line.split(' ')[1]
                bin_vector = [int(x) for x in inter_binv]
                features.append(np.array(bin_vector))
    else:
        print('{} : File does not Exist!'.format(filepath))
        exit(1)

    return np.array(features), orig_dict


def make_unweighted_graph(data, threshold):
    '''
    Take feature vectors and make unweighted graph.

    :param data: Feature vectors
    :return: Networkx graph
    '''
    G = nx.Graph()

    G.add_nodes_from(list(range(len(data))))

    for (i, j) in combinations(G.nodes, 2):
        npsum = np.sum(data[i] == data[j])
        if npsum >= threshold:
            G.add_edge(i, j)

    return G


def calc_adjmat(g):
    '''
    Calculate a graph's unweighted adjacency matrix.

    :param g: Networkx graph
    :return: Scipy sparse matrix containing the unweighted adjacency matrix
    '''
    g_am = nx.adjacency_matrix(g, nodelist=None, weight=None)

    return g_am


def compute_mod_mat(matrix):
    '''
    Compute modularity matrix for given adjacency matrix.
    Pij -- number of expected edges between nodes i and j.
    k   -- vector containing the expected degree for all nodes.

    :param matrix: Adjacency matrix
    :return: Modularity matrix
    '''
    i = np.shape(matrix)[0]
    j = np.shape(matrix)[1]
    Pij = np.zeros((i, j))
    print('Computing Expected Degree (Ki)...')
    k = matrix.sum(axis=0)

    m = k.sum(axis=1)[0, 0]

    print('Computing Pij...')
    for row in np.arange(i):
        for col in np.arange(j):
            Pij[row, col] = (k[0, row] * k[0, col]) / m

    print('Computing Modularity Matrix...')
    Bij = matrix - Pij

    return Bij


def calc_eigen(matrix):
    '''
    Compute eigenvalues and eigenvectors of a matrix.

    :param matrix: The matrix for which eigen values need to be computed
    :return: Array containing eigen values; Array containing eigenvectors as
             columns
    '''

    eigval, eigvec = eig(matrix)
    return eigval, eigvec


def prep_cluster(eigvec):
    '''
    Assign cluster label depending on signs of the elements of the most
    positive eigenvector.

    :param eigvec: matrix containing eigenvectors
    :return: list containing cluster assignments.
    '''
    labels = []
    index = range(len(eigvec[:, 0]))
    for i in index:
        if eigvec[:, 0][i, 0] < 0:
            labels.append(1)
        elif eigvec[:, 0][i, 0] >= 0:
            labels.append(0)
    return labels


def main():
    threshold = [12, 14]

    for t in threshold:
        # Phase 1
        print('Phase 1...')
        filelist = [f for f in listdir(DATA_PH1) if isfile(DATA_PH1 + f)]
        for filename in filelist:
            features, orig_dict = preprocess(DATA_PH1 + filename)
            g = make_unweighted_graph(features, t)

            del features

            am = calc_adjmat(g)

            print('Saving undirected graph for {}'.format(filename))
            nx.write_gpickle(g, GRPH_PH1 + filename + '_t' + str(t) +
                             '_unweight.gpickle', protocol=4)
            del g

            mod = compute_mod_mat(am)

            print('Saving adjacency matrix for {}'.format(filename))
            with open(ADJM_PH1 + filename + '_t' + str(t) + '_am.pickle', 'wb') as f:
                pickle.dump(am, f, protocol=4)
            del am

            eigval, eigvec = calc_eigen(mod)

            print('Saving Modularity Matrix for {}'.format(filename))
            with open(MODM_PH1 + filename + '_t' + str(t) + '_m.pickle', 'wb') as f:
                pickle.dump(mod, f, protocol=4)
            del mod

            labels = prep_cluster(eigvec)

            print('Saving Eigenvalues and eigenvectors for {}'.format(filename))
            with open(EIGN_PH1 + filename + '_t' + str(t) +
                      '_eigval.pickle', 'wb') as f:
                pickle.dump(eigval, f, protocol=4)
            with open(EIGN_PH1 + filename + '_t' + str(t) +
                      '_eigvec.pickle', 'wb') as f:
                pickle.dump(eigvec, f, protocol=4)
            del eigvec, eigval

            vec_list = list(orig_dict.keys())

            for i in range(len(labels)):
                if labels[i] == 1:
                    with open(DATA_PH2 + str(t) + '/' + filename + '_t' + str(t) + '_c1', 'a') as f:
                        f.write(vec_list[i] + ' ' + orig_dict[vec_list[i]])
                elif labels[i] == 0:
                    with open(DATA_PH2 + str(t) + '/' + filename + '_t' + str(t) + '_c0', 'a') as f:
                        f.write(vec_list[i] + ' ' + orig_dict[vec_list[i]])
            del labels

        # Phase 2
        print('Phase 2...')
        filelist = [f for f in listdir(DATA_PH2 + str(t)) if isfile(DATA_PH2 + str(t) + '/' + f)]
        for filename in filelist:
            features, curr_dict = preprocess(DATA_PH2 + str(t) + '/' + filename)
            orig_dict = preprocess(DATA_PH1 + filename[:-7])[1]
            g = make_unweighted_graph(features, t)

            del features

            am = calc_adjmat(g)

            print('Saving undirected graph for {}'.format(filename))
            nx.write_gpickle(g, GRPH_PH2 + filename + '_t' + str(t) +
                             '_unweight.gpickle', protocol=4)
            del g

            mod = compute_mod_mat(am)

            print('Saving adjacency matrix for {}'.format(filename))
            with open(ADJM_PH2 + filename + '_t' + str(t) + '_am.pickle', 'wb') as f:
                pickle.dump(am, f, protocol=4)
            del am

            eigval, eigvec = calc_eigen(mod)

            print('Saving Modularity Matrix for {}'.format(filename))
            with open(MODM_PH2 + filename + '_t' + str(t) + '_m.pickle', 'wb') as f:
                pickle.dump(mod, f, protocol=4)
            del mod

            labels = prep_cluster(eigvec)

            print('Saving Eigenvalues and eigenvectors for {}'.format(filename))
            with open(EIGN_PH2 + filename + '_t' + str(t) +
                      '_eigval.pickle', 'wb') as f:
                pickle.dump(eigval, f, protocol=4)
            with open(EIGN_PH2 + filename + '_t' + str(t) +
                      '_eigvec.pickle', 'wb') as f:
                pickle.dump(eigvec, f, protocol=4)
            del eigvec, eigval

            vec_list = list(curr_dict.keys())

            for i in range(len(labels)):
                if labels[i] == 1:
                    with open(RSLT_PH2 + filename + '_t' + str(t) + '_c1', 'a') as f:
                        f.write(vec_list[i] + ' ' + orig_dict[vec_list[i]])
                elif labels[i] == 0:
                    with open(RSLT_PH2 + filename + '_t' + str(t) + '_c0', 'a') as f:
                        f.write(vec_list[i] + ' ' + orig_dict[vec_list[i]])


if __name__ == '__main__':
    main()
    print('Done')
    exit(0)
