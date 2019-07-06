'''
Wordvectors averaged for sentence vector
'''

import os
import corpus
from sklearn.cluster import KMeans
from gensim.models import word2vec as w2v
import numpy as np

MODELPATH = os.path.join('MODELS')
RESULTPATH = os.path.join('RESULTS')


def main():
    model = w2v.Word2Vec.load(MODELPATH + '/dataset7_w2v_6')
    data_corpus = corpus.get_corpus('dataset7', 0)
    # data_orig = corpus.get_datalist('jeu_hota1')
    senvectors = []
    for i in data_corpus.index:
        senvecs = []
        for x in data_corpus.loc[i]['sentence']:
            senvecs.append(model[x])
        senvecs = np.vstack(np.array(senvecs))
        senvecs = np.mean(senvecs, axis=1)
        senvectors.append(senvecs)

    senvectors = np.vstack(np.array(senvectors))

    kmeans = KMeans(n_clusters=4)
    kmeans.fit(senvectors)
    cl_kmeans = kmeans.predict(senvectors)
    with open(RESULTPATH + '/kmeans4_dataset7', 'w') as f:
        for l in cl_kmeans:
            f.write(str(l) + '\n')


if __name__ == '__main__':
    main()
