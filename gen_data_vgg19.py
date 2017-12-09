import os
import tensorflow as tf
import pandas as pd
import time
import numpy as np

from collections import Counter
from core.utils import *
from core.vggnet import Vgg19



#==================

batch_size = 100
vgg_model = './data/imagenet-vgg-verydeep-19.mat'

#train_caption_file = 'train.csv'
#test_caption_file = 'test.csv'

image_dir = 'image/val2014_resized'
npy_dir = 'image/trainval2014_features'

#==================


def read_file(csv_file):
    data = pd.read_csv(csv_file)
    img_list = []
    id_list = []
    for img_id, row in df.iterrows():
        ID, ext = os.path.splitext(row['img_id'])
        img_list.append(os.path.join(image_dir, 'COCO_trainval2014_{:012d}{}'.format(ID, ext)))
        id_list.append(row['img_id'])

    return id_list, img_list


def main():

    #read file
    name_list = [name for name in os.listdir(image_dir) if name.endswith('.jpg')]
    img_list = [os.path.join(image_dir, name) for name in name_list]

    num_examples = len(name_list)


    # create input batch
    input_slice = tf.train.slice_input_producer(
        [name_list, img_list], capacity=batch_size*8)
    
    img_content = tf.read_file(input_slice[1])
    name = input_slice[0]

    img = tf.image.decode_jpeg(img_content, channels=3)
    img = tf.cast(img, dtype=tf.float32)
    
    img.set_shape([224, 224, 3])

    mean = tf.constant(np.array([104., 117., 124.]), dtype=tf.float32)
    mean = tf.reshape(mean, [1, 1, 3])
    img_mean = img - mean


    name_batch, img_batch = tf.train.batch(
            [name, img_mean], 
            num_threads=16, 
            batch_size=batch_size, 
            allow_smaller_final_batch=True)


    # create resnet
    vggnet = Vgg19(vgg_model)
    vggnet.set_inputs(img_batch)
    vggnet.build()

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        sess.run(tf.global_variables_initializer())

        num_step = (num_examples + batch_size -1) // batch_size

        for step in range(num_step):
            start_time = time.time()
            n_, img_ = sess.run([name_batch, vggnet.features])

            time_elapsed = time.time() - start_time
            print('processing batch[{}/{}] {:.2f} sec/batch'.format(step+1, num_step, time_elapsed))
            for idx in range(len(n_)):
                filename = n_[idx].decode('utf-8').replace('jpg','npy').replace('train', 'trainval')
                filepath = os.path.join(npy_dir, filename)
                img_features = img_[idx, :, :]
                #print('   saving {}, shape: {}'.format(filepath, img_features.shape))
                np.save(filepath, img_features)

        print('Complete')


    coord.request_stop()
    coord.join(threads)




if __name__ == '__main__':
    main()
