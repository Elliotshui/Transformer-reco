import os
import pickle
import numpy as np
from hparams import Hparams


def seq_to_string(seq):
    s = ''
    for item in seq:
        s += str(item) + ' ' 
    return s

def write_to_file(path, seqs):
    f = open(path, 'w')
    for seq in seqs:
        f.write(seq_to_string(seq) + '\n')
    f.close()
    return

def create_movielens(path, min_len, max_len, y_len):

    f = open(path, 'rb')
    dat = pickle.load(f)
    train1 = []
    train2 = []
    valid1 = []
    valid2 = []
    test1 = []
    test2 = []
    for seq in dat['data']:
        if len(seq['train']) >= min_len:
            k = np.random.randint(min_len, max_len + 1)
            train_seq = seq['train'][-k: ]
            train1.append(train_seq[0: -y_len])
            train2.append(train_seq[-y_len: ])

            valid_seq = seq['train'][-max_len: ]
            valid1.append(valid_seq)
            valid2.append(seq['valid'])

            seq['train'].extend(seq['valid'])
            test_seq = seq['train'][-max_len: ]
            test1.append(test_seq)
            test2.append(seq['test'])

    f_vocab = open('movielens-dat/vocab', 'w')
    print(len(dat['vocab']))
    for id in dat['vocab'].keys():
        f_vocab.write(dat['vocab'][id]['word'] + ' ' + str(id) + '\n')

    write_to_file('movielens-dat/train1', train1)
    write_to_file('movielens-dat/train2', train2)
    write_to_file('movielens-dat/valid1', valid1)
    write_to_file('movielens-dat/valid2', valid2)
    write_to_file('movielens-dat/test1', test1)
    write_to_file('movielens-dat/test2', test2)


if __name__ == '__main__':
    create_movielens('movielens', 30, 50, 10)
