import numpy as np
import _pickle as pickle
from my_model import CaptionGenerator
from my_solver import CaptioningSolver

import os

#=============================
encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'

record_path = './record'
#=============================


training_records = [os.path.join(record_path, f) for f in os.listdir(record_path) if f.endswith('.tfrecord')]

def main():
    enc_map = pickle.load(open(encode_map, 'rb'))
    dec_map = pickle.load(open(decode_map, 'rb'))
    vocab_size = len(dec_map)
    model = CaptionGenerator(enc_map, dec_map, vocab_size)

    solver = CaptioningSolver(model, training_records, n_epochs=1000, batch_size=128,
                            learning_rate = 0.001, print_every=10, save_every=5, restore_model='model/lstm/', 
                            model_path='model/lstm/', log_path = 'log/')
    solver.train()


if __name__ == '__main__':
    main()    
