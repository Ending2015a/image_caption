# =========================================================================================
# Implementation of "Show, Attend and Tell: Neural Caption Generator With Visual Attention".
# There are some notations.
# N is batch size.
# L is spacial size of feature vector (196).
# D is dimension of image feature vector (512).
# T is the number of time step which is equal to caption's length-1 (16).
# V is vocabulary size (about 10000).
# M is dimension of word vector which is embedding size (default is 512).
# H is dimension of hidden state (default is 1024).
# =========================================================================================

from __future__ import division

import tensorflow as tf
import os

from tensorflow.contrib.data import Dataset, Iterator


class CaptionGenerator(object):

    def __init__(self, enc_map, dec_map, vocab_size, dim_feature=[196, 512], dim_embed=512, dim_hidden=1024, max_step=20,
                prev2out=True, ctx2out=True, alpha_c=0.0, selector=True, dropout=True, mode='train'):

        """
        Args:
            enc_map: word-to-index mapping dictionary.
            dec_map: index-to-word mapping dictionary.
            vocab_size: size of the vocabulary.
            dim_feature: (optional) Dimension of vggnet19 conv5_3 feature vectors.
            dim_embed: (optional) Dimension of word embedding.
            dim_hidden: (optional) Dimension of all hidden state.
            prev2out: (optional) previously generated word to hidden state. (see Eq (7) for explanation)
            ctx2out: (optional) context to hidden state (see Eq (7) for explanation)
            alpha_c: (optional) Doubly stochastic regularization coefficient. (see Section (4.2.1) for explanation)
            selector: (optional) gating scalar for context vector. (see Section (4.2.1) for explanation)
            dropout: (optional) If true then dropout layer is added.
        """

        self.enc_map = enc_map
        self.dec_map = dec_map
        self.prev2out = prev2out
        self.ctx2out = ctx2out
        self.alpha_c = alpha_c
        self.selector = selector
        self.dropout = dropout
        self.V = vocab_size
        self.L = dim_feature[0]
        self.D = dim_feature[1]
        self.M = dim_embed
        self.H = dim_hidden
        self.T = max_step
        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.const_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        

        self._start = 0
        self._null = 1
        self.mode = mode

    def create_dataset(self, batch_size):
        def training_parser(record):
            keys_to_features = {
                'id': tf.FixedLenFeature([1], tf.int64),
                'feature': tf.FixedLenFeature([self.L*self.D], tf.float32),
                #'caption': tf.FixedLenFeature([21], dtype=tf.int64),
                'padded': tf.FixedLenFeature([21], dtype=tf.int64)
                }
            features = tf.parse_single_example(record, features=keys_to_features)
            file_id = features['id']
            feature = features['feature']
            padded = features['padded']
            padded = tf.cast(padded, tf.int32)

            caption_len = tf.shape(padded)[0]
            input_len = tf.expand_dims(tf.subtract(caption_len, 1), 0)

            input_seq = tf.slice(padded, [0], input_len)
            output_seq = tf.slice(padded, [1], input_len)

            input_seq.set_shape([20])
            output_seq.set_shape([20])


            mask = tf.ones(input_len, dtype=tf.int32)


            records = {
                'feature': feature,
                'padded': padded,
                'input_seq': input_seq,
                'output_seq': output_seq,
                'mask': mask}
            return records


        def tfrecord_iterator(filenames, batch_size, record_parser):
            dataset = tf.data.TFRecordDataset(filenames)
            dataset = dataset.map(record_parser, num_parallel_calls=16)

            dataset = dataset.repeat()
            dataset = dataset.shuffle(batch_size*3)

            dataset = dataset.padded_batch(
                    batch_size=batch_size,
                    padded_shapes={
                        'feature': [None],
                        'padded': [None],
                        'input_seq': [None],
                        'output_seq': [None],
                        'mask': [None]},
                    padding_values={
                        'feature': 0.0,
                        'padded': 1,
                        'input_seq': 1,
                        'output_seq': 1,
                        'mask': 0}
                    )

            iterator = dataset.make_initializable_iterator()
            output_types = dataset.output_types
            output_shapes = dataset.output_shapes

            return iterator, output_types, output_shapes
        
        if self.mode == 'train':
            self.filenames = tf.placeholder(tf.string, shape=[None], name='filenames')
            self.training_iterator, types, shapes = tfrecord_iterator(self.filenames, batch_size, training_parser)

            self.handle = tf.placeholder(tf.string, shape=[], name='handle')
            iterator = tf.data.Iterator.from_string_handle(self.handle, types, shapes)
            records = iterator.get_next()

            image_embed = records['feature']
            image_embed.set_shape([None, self.L*self.D])
            image_embed = tf.reshape(image_embed, [-1, self.L, self.D])
            #image_file = records['filename']
            caption = records['padded']
            input_seq = records['input_seq']
            target_seq = records['output_seq']
            input_mask = records['mask']
        else:
            image_embed = tf.placeholder(tf.float32, shape=[None, self.L, self.D], name='image_embed')
            #image_file = tf.placeholder(tf.string, shape=[None], name='image_file')
            caption = tf.placeholder(tf.int32, shape=[None], name='input_feed')
            input_seq = tf.expand_dims(caption, axis=1)
            target_seq = None
            input_mask = None

        self.features = image_embed
        self.captions = caption
        #self.image_file = image_file
        self.input_seq = input_seq
        self.target_seq = target_seq
        self.input_mask = input_mask


    def _get_initial_lstm(self, features):
        with tf.variable_scope('initial_lstm'):
            features_mean = tf.reduce_mean(features, 1)

            w_h = tf.get_variable('w_h', [self.D, self.H], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.H], initializer=self.const_initializer)
            h = tf.nn.tanh(tf.matmul(features_mean, w_h) + b_h)

            w_c = tf.get_variable('w_c', [self.D, self.H], initializer=self.weight_initializer)
            b_c = tf.get_variable('b_c', [self.H], initializer=self.const_initializer)
            c = tf.nn.tanh(tf.matmul(features_mean, w_c) + b_c)
            return c, h

    def _word_embedding(self, inputs, reuse=False):
        with tf.variable_scope('word_embedding', reuse=reuse):
            w = tf.get_variable('w', [self.V, self.M], initializer=self.emb_initializer)
            x = tf.nn.embedding_lookup(w, inputs, name='word_vector')  # (N, T, M) or (N, M)
            
            return x

    def _project_features(self, features):
        with tf.variable_scope('project_features'):
            w = tf.get_variable('w', [self.D, self.D], initializer=self.weight_initializer)
            features_flat = tf.reshape(features, [-1, self.D])
            features_proj = tf.matmul(features_flat, w)  
            features_proj = tf.reshape(features_proj, [-1, self.L, self.D])
            return features_proj

    def _attention_layer(self, features, features_proj, h, reuse=False):
        with tf.variable_scope('attention_layer', reuse=reuse):
            w = tf.get_variable('w', [self.H, self.D], initializer=self.weight_initializer)
            b = tf.get_variable('b', [self.D], initializer=self.const_initializer)
            w_att = tf.get_variable('w_att', [self.D, 1], initializer=self.weight_initializer)

            h_att = tf.nn.relu(features_proj + tf.expand_dims(tf.matmul(h, w), 1) + b)    # (N, L, D)
            out_att = tf.reshape(tf.matmul(tf.reshape(h_att, [-1, self.D]), w_att), [-1, self.L])   # (N, L)
            alpha = tf.nn.softmax(out_att)  
            context = tf.reduce_sum(features * tf.expand_dims(alpha, 2), 1, name='context')   #(N, D)
            return context, alpha
  
    def _selector(self, context, h, reuse=False):
        with tf.variable_scope('selector', reuse=reuse):
            w = tf.get_variable('w', [self.H, 1], initializer=self.weight_initializer)
            b = tf.get_variable('b', [1], initializer=self.const_initializer)
            beta = tf.nn.sigmoid(tf.matmul(h, w) + b, 'beta')    # (N, 1)
            context = tf.multiply(beta, context, name='selected_context') 
            return context, beta
  
    def _decode_lstm(self, x, h, context, dropout=False, reuse=False):
        with tf.variable_scope('logits', reuse=reuse):
            w_h = tf.get_variable('w_h', [self.H, self.M], initializer=self.weight_initializer)
            b_h = tf.get_variable('b_h', [self.M], initializer=self.const_initializer)
            w_out = tf.get_variable('w_out', [self.M, self.V], initializer=self.weight_initializer)
            b_out = tf.get_variable('b_out', [self.V], initializer=self.const_initializer)

            if dropout:
                h = tf.nn.dropout(h, 0.5)
            h_logits = tf.matmul(h, w_h) + b_h

            if self.ctx2out:
                w_ctx2out = tf.get_variable('w_ctx2out', [self.D, self.M], initializer=self.weight_initializer)
                h_logits += tf.matmul(context, w_ctx2out)

            if self.prev2out:
                h_logits += x
            h_logits = tf.nn.tanh(h_logits)

            if dropout:
                h_logits = tf.nn.dropout(h_logits, 0.5)
            out_logits = tf.matmul(h_logits, w_out) + b_out
            return out_logits
        
    def _batch_norm(self, x, mode='train', name=None):
        return tf.contrib.layers.batch_norm(inputs=x, 
                                            decay=0.95,
                                            center=True,
                                            scale=True,
                                            is_training=(mode=='train'),
                                            updates_collections=None,
                                            scope=(name+'batch_norm'))

    def build_model(self):
        features = self.features
        captions = self.captions
        batch_size = tf.shape(features)[0]

        captions_in = self.input_seq
        captions_out = self.target_seq
        mask = self.input_mask

        captions_in.set_shape([None, self.T])
        captions_out.set_shape([None, self.T])
        mask.set_shape([None, self.T])

        mask = tf.cast(mask, dtype=tf.float32)

        onehot = tf.one_hot(captions_out, self.V)
        
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='train', name='conv_features')
        
        c, h = self._get_initial_lstm(features=features)
        #x = self._word_embedding(inputs=captions_in)
        features_proj = self._project_features(features=features)

        losses = []
        alpha_list = []
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H, initializer=tf.random_normal_initializer(stddev=0.03))

        for t in range(self.T):
            word_emb = self._word_embedding(inputs=captions_in[:, t], reuse=tf.AUTO_REUSE) # (batch_size, dim_embed)

            context, alpha = self._attention_layer(features, features_proj, h, reuse=tf.AUTO_REUSE)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=tf.AUTO_REUSE) 

            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                _, (c, h) = lstm_cell(inputs=tf.concat([word_emb, context], 1), state=[c, h])

            logits = self._decode_lstm(word_emb, h, context, dropout=self.dropout, reuse=tf.AUTO_REUSE)

            cross_entropy = tf.multiply(tf.nn.softmax_cross_entropy_with_logits(labels=onehot[:,t,:], logits=logits), mask[:, t])
            losses.append(tf.reduce_sum(cross_entropy))

        loss = tf.reduce_sum(losses)
           
        if self.alpha_c > 0:
            alphas = tf.transpose(tf.pack(alpha_list), (1, 0, 2))     # (N, T, L)
            alphas_all = tf.reduce_sum(alphas, 1)      # (N, L)
            alpha_reg = self.alpha_c * tf.reduce_sum((16./196 - alphas_all) ** 2)     
            loss += alpha_reg

        return loss / tf.to_float(batch_size)

    def build_sampler(self, max_len=20):
        features = self.features
        
        # batch normalize feature vectors
        features = self._batch_norm(features, mode='test', name='conv_features')
        
        c, h = self._get_initial_lstm(features=features)
        features_proj = self._project_features(features=features)

        sampled_word_list = []
        alpha_list = []
        beta_list = []
        lstm_cell = tf.contrib.rnn.LSTMCell(num_units=self.H)

        for t in range(max_len):
            if t == 0:
                x = self._word_embedding(inputs=tf.fill([tf.shape(features)[0]], self._start))
            else:
                x = self._word_embedding(inputs=sampled_word, reuse=tf.AUTO_REUSE)  
          
            context, alpha = self._attention_layer(features, features_proj, h, reuse=tf.AUTO_REUSE)
            alpha_list.append(alpha)

            if self.selector:
                context, beta = self._selector(context, h, reuse=tf.AUTO_REUSE) 
                beta_list.append(beta)

            with tf.variable_scope('lstm', reuse=tf.AUTO_REUSE):
                _, (c, h) = lstm_cell(inputs=tf.concat([x, context], 1), state=[c, h])

            logits = self._decode_lstm(x, h, context, reuse=tf.AUTO_REUSE)
            sampled_word = tf.argmax(logits, 1)       
            sampled_word_list.append(sampled_word)     

        alphas = tf.transpose(tf.stack(alpha_list), (1, 0, 2))     # (N, T, L)
        betas = tf.transpose(tf.squeeze(beta_list), (1, 0))    # (N, T)
        sampled_captions = tf.transpose(tf.stack(sampled_word_list), (1, 0))     # (N, max_len)
        return alphas, betas, sampled_captions
