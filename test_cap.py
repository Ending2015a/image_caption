import numpy as np
import _pickle as pickle
from my_model import CaptionGenerator
from my_solver import CaptioningSolver

import os

#=============================
encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'

record_path = './record'
test_file = './test_dict.npy'
#=============================


def main():
    enc_map = pickle.load(open(encode_map, 'rb'))
    dec_map = pickle.load(open(decode_map, 'rb'))
    vocab_size = len(dec_map)
    model = CaptionGenerator(enc_map, dec_map, vocab_size)

    solver = CaptioningSolver(model, None, restore_model='model/lstm/', model_path='model/lstm/', log_path = 'log/')
    solver.test(test_file)


if __name__ == '__main__':
    main()    
