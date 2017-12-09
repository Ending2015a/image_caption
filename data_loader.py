import numpy as np
import pandas as pd
import _pickle as pickle
import time
import os


encode_map = 'enc_map.pkl'
decode_map = 'dec_map.pkl'

def preprocess_coco_data(csv_file):
    df = pd.read_csv(csv_file)
    enc_map = pickle.load(open(encode_map, 'rb'))

    data = {}
    data['img_id'] = []
    data['img_file'] = []
    data['caption'] = []
    for img_id, row in df.iterrows():
        ID, ext = os.path.splitext(row['img_id'])
        data['img_id'].append(row['img_id'])
        data['img_file'].append('COCO_trainval2014_{:012d}'.format(int(ID)))
        data['caption'].append(encode_caption(enc_map, row['caption']))

    return data

def load_coco_data(csv_file):
    data = np.load(csv_file).item()
    return data

def encode_caption(enc_map, caps):
    icap = [enc_map[x] for x in caps.split(' ')]
    icap.insert(0, enc_map['<ST>'])
    icap.append(enc_map['<ED>'])
    return icap

def decode_caption(dec_map, ids):
    return ' '.join([dec_map[x] for x in ids])


if __name__ == '__main__':
    data = preprocess_coco_data('train.csv')
    np.save('enc_train_dict.npy', data)

    data = np.load('enc_train_dict.npy').item()
    
    dec_map = pickle.load(open('dec_map.pkl', 'rb'))

    for idx in range(8):
        print('{}: {} | {} | {}'.format(
                idx, data['img_id'][idx], data['img_file'][idx], 
                decode_caption(dec_map, data['caption'][idx])))

