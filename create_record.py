
import tensorflow as tf
import numpy as np
from data_loader import *
import _pickle as pickle
import time

#=================================================

feature_root = './image/trainval2014_features/' # root path where stored all feature maps in .npy format
record_file = './record/train'
split_count = 150  # split tfrecord to how many parts
num_worker = 10  # threads

data_file = './enc_train_dict.npy'
decode_map_file = './dec_map.pkl'

caption_padding_size = 21

#=================================================

# read decode map (index->word)
dec_map = pickle.load(open(decode_map_file, 'rb'))



def create_tfrecords(data, record_name, split_num=100, start_from=0):

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    # pad or crop captions to fixed size with [padding_word]
    def padding_caption(caption, size, padding_word=1):
        length = len(caption)
        crop = max(length-size, 0)
        pad = max(size-length, 0)
        caption = caption[0:length-crop] + [padding_word] * pad
        caption[-1] = 1
        return caption

    file_ids = data['img_id']
    file_names = data['img_file']
    captions = data['caption']

    num_records_per_file = len(file_ids) // split_num

    total_count = 0
    print('craete tfrecord.....')

    def decode(ids):
        return ' '.join([dec_map[x] for x in ids])

    def pack_task(record_num):
        count = 0
        writer = tf.python_io.TFRecordWriter('{}-part{}.tfrecord'.format(record_name, record_num+1))

        st = record_num * num_records_per_file
        ed = (record_num+1) * num_records_per_file if record_num != split_num -1 else len(file_ids)

        start_time = time.time()

        for idx in range(st, ed):
            # read feature maps (default size is 196*512 from vgg19 conv5_3)
            img_feature = np.load(os.path.join(feature_root, file_names[idx] + '.npy'))
            # flatten to 1D array
            img_feature = img_feature.reshape(-1)
            # caption
            cap = captions[idx]
            # file id
            file_id = int(file_ids[idx].split('.')[0])
            # padded caption
            padded_cap = padding_caption(cap, caption_padding_size)

            # create examples
            example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'id': _int64_feature([filename_id]),  # 1
                            'feature': _float_feature(img_feature),  # 196*512
                            'caption': _int64_feature(cap), # variable size
                            'padded': _int64_feature(padded_cap) # caption_padding_size
                        }))

            count += 1
            writer.write(example.SerializeToString())
            if(idx+1) % 1000 == 0:
                print('[record {}] id: {}, feature shape: {}, caption: {}, padded: {}'.format(idx,
                            filename_id, img_feature.shape, decode(caps), decode(padded_caps)))

        elapsed_time = time.time() - start_time
        remaining_time = float((split_num-(record_num+1)) // num_worker) * elapsed_time
        print('create {}-{}.tfrecord -- contains {} records / in {:4f} sec / remaining about {:4f} sec'
                    .format(record_name, str(record_num+1), count, elapsed_time, remaining_time)
        
        total_count += count
        writer.close()

    # start task
    for i in range(start_from, num_files):
        pack_task(i)

    print('Total records: {}'.format(total_count))


if __name__ == '__main__':
    data = load_coco_data(data_file)
    create_tfrecords(data, record_file, split_count)
