from gensim.models import word2vec as w2v
import os
import itertools as it
import corpus

DATAPATH = os.path.join('DATA')
MODELPATH = os.path.join('MODELS')


def compute_vectors(filename, data_pos):
    data = corpus.get_corpus(filename, data_pos)
    sentences = it.chain(sent for sent in data.loc[:]['sentence'])
    print('Computing word vectors for {}...'.format(filename))
    wvmodel = w2v.Word2Vec(sentences=sentences, sg=1, size=6,
                           hs=1, min_count=1, workers=1, iter=50000,
                           compute_loss=True, window=6, seed=123, )
    print('Saving Model...')
    wvmodel.save(MODELPATH + '/' + filename + '_w2v_6_1')
    return 0


def main():
    '''
    data_pos = 0 for jeu_hota1, jeu_hota2
           1 for jeu_hota3
           2 for jeu_hota4
    '''
    compute_vectors('jeu_hota1', 0)
    compute_vectors('jeu_hota2', 0)
    compute_vectors('jeu_hota3', 1)
    compute_vectors('jeu_hota4', 2)
    print('Done')


if __name__ == '__main__':
    main()
