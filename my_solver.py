import tensorflow as tf
import matplotlib.pyplot as plt
import skimage.transform
import numpy as np
import time
import os 
import _pickle as pickle
from scipy import ndimage
#from utils import *
#from bleu import evaluate


class CaptioningSolver(object):
    def __init__(self, model, data, **kwargs):
        """
        Required Arguments:
            - model: Show Attend and Tell caption generating model
            - data: Training data; dictionary with the following keys:
                - features: Feature vectors of shape (82783, 196, 512)
                - file_names: Image file names of shape (82783, )
                - captions: Captions of shape (400000, 17) 
                - image_idxs: Indices for mapping caption to image of shape (400000, ) 
                - word_to_idx: Mapping dictionary from word to index 
            - val_data: validation data; for print out BLEU scores for each epoch.
        Optional Arguments:
            - n_epochs: The number of epochs to run for training.
            - batch_size: Mini batch size.
            - update_rule: A string giving the name of an update rule
            - learning_rate: Learning rate; default value is 0.01.
            - print_every: Integer; training losses will be printed every print_every iterations.
            - save_every: Integer; model variables will be saved every save_every epoch.
            - pretrained_model: String; pretrained model path 
            - model_path: String; model path for saving 
            - test_model: String; model path for test 
        """

        self.model = model
        self.data = data
        self.n_epochs = kwargs.pop('n_epochs', 500)
        self.batch_size = kwargs.pop('batch_size', 100)
        self.update_rule = kwargs.pop('update_rule', 'adam')
        self.learning_rate = kwargs.pop('learning_rate', 0.01)
        self.print_bleu = kwargs.pop('print_bleu', False)
        self.print_every = kwargs.pop('print_every', 100)
        self.save_every = kwargs.pop('save_every', 1)
        self.log_path = kwargs.pop('log_path', './log/')
        self.model_path = kwargs.pop('model_path', './model/')
        self.restore_model = kwargs.pop('restore_model', None)
        self.test_model = kwargs.pop('test_model', './model/lstm/model-1')
        self.record_size = kwargs.pop('record_size', 513)

        # set an optimizer by update rule
        if self.update_rule == 'adam':
            self.optimizer = tf.train.AdamOptimizer
        elif self.update_rule == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer
        elif self.update_rule == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer

        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)


    def train(self):

        global_step = tf.get_variable('global_step', [],
                                    initializer=tf.constant_initializer(0), trainable=False)
        # train/val dataset
        n_examples = self.record_size*len(self.data)
        n_iters_per_epoch = int(np.ceil(float(n_examples)/self.batch_size))

        # build graphs for training model and sampling captions
        self.model.create_dataset(self.batch_size)

        with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
            loss = self.model.build_model()
            _, _, generated_captions = self.model.build_sampler(max_len=20)

        # train op
        with tf.name_scope('optimizer'):
            optimizer = self.optimizer(learning_rate=self.learning_rate)
            grads = optimizer.compute_gradients(loss)
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name+'/gradient', grad)
            train_op = optimizer.apply_gradients(grads, global_step=global_step)
           
        # summary op
        tf.summary.scalar('batch_loss', loss)
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        
        summary_op = tf.summary.merge_all() 

        print("The number of epoch: %d" %self.n_epochs)
        print("Data size: %d" %n_examples)
        print("Batch size: %d" %self.batch_size)
        print("Iterations per epoch: %d" %n_iters_per_epoch)
        
        config = tf.ConfigProto(allow_soft_placement = True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            summary_writer = tf.summary.FileWriter(self.log_path, graph=tf.get_default_graph())
            saver = tf.train.Saver(max_to_keep=10)

            # restore latest pretrained model, if param resotre_model has been assigned model path
            if self.restore_model is not None:
                latest_ckpt = tf.train.latest_checkpoint(self.restore_model)
                if not latest_ckpt:
                    print('Not found checkpoint in {}'.format(self.restore_model))
                else:
                    print("Start training with pretrained Model {} ...".format(latest_ckpt))
                    saver.restore(sess, latest_ckpt)

            prev_loss = -1
            curr_loss = 0

            training_handle = sess.run(self.model.training_iterator.string_handle())
            sess.run(self.model.training_iterator.initializer, feed_dict={self.model.filenames: self.data})

            start_t = time.time()

            try:
                for e in range(self.n_epochs):
                    for i in range(n_iters_per_epoch):
                
                        feed_dict = {self.model.handle: training_handle}
                        op = [global_step, loss, train_op, self.model.captions, generated_captions]
                        step_, l, _, ground_truths, gen_caps = sess.run(op, feed_dict=feed_dict)
                        curr_loss += l

                        # write summary for tensorboard visualization
                        if i % 10 == 0:
                            summary = sess.run(summary_op, feed_dict)
                            summary_writer.add_summary(summary, global_step=step_)

                        if (i+1) % self.print_every == 0:
                            print("[epoch {} / step {}] loss: {:.5f}".format(e+1, step_, l))
                            ground_truths = ground_truths[0]

                            def decode(ids):
                                return ' '.join([self.model.dec_map[x] for x in ids])

                            ground_truths = decode(ground_truths)

                            print('    Ground truth: {}'.format(ground_truths))
                            gen_caps = gen_caps[0]
                            gen_caps = decode(gen_caps)
                            print("    Generated caption: {}".format(gen_caps))

                    #print("  Previous epoch loss: ", prev_loss)
                    #print("  Current epoch loss: ", curr_loss)
                    print("  [epoch {}] End. prev loss: {:.5f}, cur loss: {:.5f}, Elapsed time: {:.4f}".format(
                                            e+1, prev_loss, curr_loss, time.time() - start_t))
                    prev_loss = curr_loss
                    curr_loss = 0
                    '''
                    # print out BLEU scores and file write
                    if self.print_bleu:
                        all_gen_cap = np.ndarray((val_features.shape[0], 20))
                        for i in range(n_iters_val):
                            features_batch = val_features[i*self.batch_size:(i+1)*self.batch_size]
                            feed_dict = {self.model.features: features_batch}
                            gen_cap = sess.run(generated_captions, feed_dict=feed_dict)  
                            all_gen_cap[i*self.batch_size:(i+1)*self.batch_size] = gen_cap
                        
                        all_decoded = decode_captions(all_gen_cap, self.model.idx_to_word)
                        save_pickle(all_decoded, "./data/val/val.candidate.captions.pkl")
                        scores = evaluate(data_path='./data', split='val', get_scores=True)
                        write_bleu(scores=scores, path=self.model_path, epoch=e)
                    '''
                    # save model's parameters
                    if (e+1) % self.save_every == 0:
                        saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                        print("model-%s saved." % (step_))
            
            except KeyboardInterrupt:
                print('Interrupt !!')
                saver.save(sess, os.path.join(self.model_path, 'model'), global_step=global_step)
                print('model-%s saved.' % (step_))            
            
    """
    def test(self, data, split='train', attention_visualization=False, save_sampled_captions=True):
        '''
        Args:
            - data: dictionary with the following keys:
            - features: Feature vectors of shape (5000, 196, 512)
            - file_names: Image file names of shape (5000, )
            - captions: Captions of shape (24210, 17) 
            - image_idxs: Indices for mapping caption to image of shape (24210, ) 
            - features_to_captions: Mapping feature to captions (5000, 4~5)
            - split: 'train', 'val' or 'test'
            - attention_visualization: If True, visualize attention weights with images for each sampled word. (ipthon notebook)
            - save_sampled_captions: If True, save sampled captions to pkl file for computing BLEU scores.
        '''

        features = data['features']

        # build a graph to sample captions
        alphas, betas, sampled_captions = self.model.build_sampler(max_len=20)    # (N, max_len, L), (N, max_len)
        
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver = tf.train.Saver()

            if self.restore_model is None:
                print('please specify the restore path')
                return
            
            latest_ckpt = tf.train.latest_checkpoint(self.restore_model)
            if not latest_ckpt:
                print('Not found checkpoint in {}'.format(self.restore_model))
                return
            else:
                print('Start testing with pretrained Model {}...'.format(latest_ckpt))
                saver.restore(sess, latest_ckpt)
            


            #features_batch, image_files = sample_coco_minibatch(data, self.batch_size)
            feed_dict = { self.model.features: features_batch }
            alps, bts, sam_cap = sess.run([alphas, betas, sampled_captions], feed_dict)  # (N, max_len, L), (N, max_len)
            
            def decode_from_st_to_ed(ids, st=0, ed=1):
                start_point = 0 if st not in ids else ids.index(st)
                end_point = None if ed not in ids else ids.index(ed)
                return ' '.join([self.model.dec_map[x] for x in ids[start_point:end_point])

            decoded = decode_from_st_to_ed(sam_cap)

            if attention_visualization:
                for n in range(10):
                    print "Sampled Caption: %s" %decoded[n]

                    # Plot original image
                    img = ndimage.imread(image_files[n])
                    plt.subplot(4, 5, 1)
                    plt.imshow(img)
                    plt.axis('off')

                    # Plot images with attention weights 
                    words = decoded[n].split(" ")
                    for t in range(len(words)):
                        if t > 18:
                            break
                        plt.subplot(4, 5, t+2)
                        plt.text(0, 1, '%s(%.2f)'%(words[t], bts[n,t]) , color='black', backgroundcolor='white', fontsize=8)
                        plt.imshow(img)
                        alp_curr = alps[n,t,:].reshape(14,14)
                        alp_img = skimage.transform.pyramid_expand(alp_curr, upscale=16, sigma=20)
                        plt.imshow(alp_img, alpha=0.85)
                        plt.axis('off')
                    plt.show()

            if save_sampled_captions:
                all_sam_cap = np.ndarray((features.shape[0], 20))
                num_iter = int(np.ceil(float(features.shape[0]) / self.batch_size))
                for i in range(num_iter):
                    features_batch = features[i*self.batch_size:(i+1)*self.batch_size]
                    feed_dict = { self.model.features: features_batch }
                    all_sam_cap[i*self.batch_size:(i+1)*self.batch_size] = sess.run(sampled_captions, feed_dict)  
                all_decoded = decode_captions(all_sam_cap, self.model.idx_to_word)
                save_pickle(all_decoded, "./data/%s/%s.candidate.captions.pkl" %(split,split))
    """


