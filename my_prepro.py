import os
import numpy as np
import pandas as pd
import tensorflow as tf
import _pickle as cPickle


vocab = cPickle.load(open('vocab.pkl', 'rb'))
print('total {} vocabularies'.format(len(vocab)))

def count_vocab_occurance(vocab, df):
    voc_cnt = {v: 0 for v in vocab}
    for img_id, row in df.iterrows():
        for w in row['caption'].split(' '):
            voc_cnt[w] += 1
    return voc_cnt

df_train = pd.read_csv('train.csv')

print('count vicabulary occurances...')
voc_cnt = count_vocab_occurance(vocab, df_train)

thrhd = 50
x = np.array(list(voc_cnt.values()))
print('{} words appear >= 50 times'.format(np.sum(x[(-x).argsort()] >= thrhd)))



# In [4]:


def build_voc_mapping(voc_cnt, thrhd):
    def add(enc_map, dec_map, voc):
        enc_map[voc] = len(dec_map) 
        dec_map[len(dec_map)] = voc
        return enc_map, dec_map

    enc_map, dec_map = {}, {}
    for voc in ['<ST>', '<ED>', '<RARE>']:
        enc_map, dec_map = add(enc_map, dec_map, voc)
    for voc, cnt in voc_cnt.items():
        if cnt < thrhd:
            enc_map[voc] = enc_map['<RARE>']

        else:
            enc_map, dec_map = add(enc_map, dec_map, voc)
    return enc_map, dec_map

enc_map, dec_map = build_voc_mapping(voc_cnt, thrhd)
cPickle.dump(enc_map, open('enc_map.pkl', 'wb'))
cPickle.dump(dec_map, open('dec_map.pkl', 'wb'))



def caption_to_ids(enc_map, df):
    img_ids, caps = [], []
    for idx, row in df.iterrows():
        icap = [enc_map[x] for x in row['caption'].split(' ')]
        icap.insert(0, enc_map['<ST>'])
        icap.append(enc_map['<ED>'])
        img_ids.append(row['img_id'])
        caps.append(icap)

    return pd.DataFrame({'img_id': img_ids, 'caption': caps}).set_index(['img_id'])

enc_map = cPickle.load(open('enc_map.pkl', 'rb'))
print('[transform captions into sequences of IDs]...')
df_proc = caption_to_ids(enc_map, df_train)
df_proc.to_csv('train_enc_cap.csv')


df_cap = pd.read_csv('train_enc_cap.csv')
enc_map = cPickle.load(open('enc_map.pkl', 'rb'))
dec_map = cPickle.load(open('dec_map.pkl', 'rb'))

vocab_size = len(dec_map)

def decode(dec_map, ids):
    return ' '.join([dec_map[x] for x in ids])

print('decoding the encoded captions back...\n')
for idx, row in df_cap.iloc[:8].iterrows():
    print('{}: {}'.format(idx, decode(dec_map, eval(row['caption']))))



