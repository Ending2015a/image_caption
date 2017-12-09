
import tensorflow as tf
import numpy as np
import cv2
from data_loader import *
import _pickle as pickle
import time

feature_root = './image/trainval2014_features/'
image_root = './image/trainval2014_resized/'

def load_image(addr):
    img = cv2.imread(addr)
    img = img.astype(np.float32)
    return img

def padded_caption(caption, size, padding=1):
    length = len(caption)
    crop = max(length-size, 0)
    pad = max(size-length, 0)

    caption = caption[0:length-crop] + [padding] * pad
    caption[-1] = 1
    return caption
    


def create_tfrecords(fid, file_id, caption, filename, num_files=100, start_from=0):

    def _float_feature(value):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

    def _int64_feature(value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def _bytes_feature(value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    num_records_per_file = len(file_id) // num_files

    total_count = 0

    print('create training datdaset....')

    dec_map = pickle.load(open('dec_map.pkl', 'rb'))

    def decode(ids):
        return ' '.join([dec_map[x] for x in ids])

    for i in range(start_from, num_files):
        count = 0
        writer = tf.python_io.TFRecordWriter(filename+'-'+str(i+1)+'.tfrecord')

        st = i*num_records_per_file
        ed = (i+1) * num_records_per_file if i != num_files - 1 else len(file_id)

        start_time = time.time()

        for idx in range(st, ed):
            img_feature = np.load(os.path.join(feature_root, file_id[idx]+'.npy'))
            img_feature = img_feature.reshape(-1)
            caps = caption[idx]
            filename_id = int(fid[idx].split('.')[0])
            padded_caps = padded_caption(caption[idx], 21)
            fn = str(os.path.join(image_root, file_id[idx]))
            example = tf.train.Example(features=tf.train.Features(
                    feature={
                        'id': _int64_feature([filename_id]),
                        'feature': _float_feature(img_feature),
                        'caption': _int64_feature(caps),
                        'padded': _int64_feature(padded_caps)
                    }))
        
            count += 1
            writer.write(example.SerializeToString())
            if (idx+1) % 1000 == 0:
                print('record {}:: fileid: {}, feature shape: {}, caption: {}, padded: {}'.format(idx,
                            filename_id, img_feature.shape, decode(caps), decode(padded_caps)))
        
        elapsed_time = time.time() - start_time
        print('create {}-{}.tfrecord -- contains {} records in {:.4f} sec'.format(filename, str(i+1), count, elapsed_time))
        total_count += count
        writer.close()

    print('Total records: {}'.format(total_count))


if __name__ == '__main__':
    data = load_coco_data('enc_train_dict.npy')
    create_tfrecords(data['img_id'], data['img_file'], data['caption'], 'record/train', 150)
