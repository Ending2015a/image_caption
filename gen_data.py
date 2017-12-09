import os
import tensorflow as tf
import _pickle as cPickle
import pandas as pd
import time
import numpy as np



from resnet50.ResNet50 import ResNet50


#==================

batch_size = 20
resnet_model = './resnet50/ResNet-50-model.npy'

#train_caption_file = 'train.csv'
#test_caption_file = 'test.csv'

image_dir = 'image/train2014_resized'

#==================


def read_file(csv_file):
    data = pd.read_csv(csv_file)
    img_list = []
    id_list = []
    for img_id, row in df.iterrows():
        ID, ext = os.path.splitext(row['img_id'])
        img_list.append(os.path.join(image_dir, 'COCO_train2014_{:012d}{}'.format(ID, ext)))
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
    net = ResNet50({'data': img_batch})

    features = net.layers['res4f']

    with tf.Session() as sess:

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        net.load(data_path=resnet_model, session=sess)

        sess.run(tf.global_variables_initializer())

        num_step = 1 #(num_examples + batch_size -1) // batch_size

        for step in range(num_step):
            start_time = time.time()
            n_, img_ = sess.run([name_batch, features])

            time_elapsed = time.time() - start_time
            print('processing batch[{}/{}] {:.2f} sec/batch'.format(step+1, num_step, time_elapsed))
            print('n_: {}'.format(n_))
            print('img_.shape: {}'.format(img_.shape))

        print('complete')


    coord.request_stop()
    coord.join(threads)




if __name__ == '__main__':
    main()
