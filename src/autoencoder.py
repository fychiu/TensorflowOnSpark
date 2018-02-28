import tensorflow as tf
import numpy as np
import os

def test(testData, classifier):
    cost = []
    batchsize = 50
    for data in DataBatch(testData,batchsize):
        cost.append(classifier(data))
    return np.mean(cost)

def DataBatch(testData, batchsize, shuffle=True):
    n = testData.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield testData[inds]

class Autoencoder(object):
    
    def __init__(self, n_hidden_layers, feat_size, learning_rate, n_frame, b_size=40, pretrain_epoch=100, load=True):
        self.n_layer_1, self.n_layer_2, self.n_layer_3 = n_hidden_layers
        self.n_layer = len(n_hidden_layers)
        self.layers = n_hidden_layers
        self.n_repr = self.n_layer_3
        self.feat_size = feat_size
        self.learning_rate = learning_rate
        self.n_frame = n_frame
        self.b_size = b_size
        self.pretrain_n_epoch = pretrain_epoch
        self.act_fn = tf.tanh
        self.load = load
    
    def dropout(self, X, rate=0.1, shape=None, mode=True):
        return tf.layers.dropout(X, rate=rate, noise_shape=shape, seed=7, training=mode)

    def initPlaceholders(self):
        self.feat_shape = [None, self.n_frame, self.feat_size]
        self.feat_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
        
    def initParams(self):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=7)
            shape_1 = [self.feat_size * self.n_frame, self.n_layer_1]
            shape_2 = [self.n_layer_1, self.n_layer_2]
            shape_3 = [self.n_layer_2, self.n_layer_3]
            self.W_1 = tf.get_variable('W_1', shape=shape_1, 
                                       initializer=initializer, trainable=True)
            self.W_2 = tf.get_variable('W_2', shape=shape_2, 
                                       initializer=initializer, trainable=True)
            self.W_3 = tf.get_variable('W_3', shape=shape_3, 
                                       initializer=initializer, trainable=True)
            self.b_1 = tf.get_variable('b_1', shape=[1, self.n_layer_1], 
                                       initializer=initializer, trainable=True)
            self.b_2 = tf.get_variable('b_2', shape=[1, self.n_layer_2], 
                                       initializer=initializer, trainable=True)
            self.b_3 = tf.get_variable('b_3', shape=[1, self.n_layer_3], 
                                       initializer=initializer, trainable=True)
            self.b_4 = tf.get_variable('b_4', shape=[1, self.n_layer_2], 
                                       initializer=initializer, trainable=True)
            self.b_5 = tf.get_variable('b_5', shape=[1, self.n_layer_1], 
                                       initializer=initializer, trainable=True)
            self.b_6 = tf.get_variable('b_6', shape=[1, self.feat_size * self.n_frame], 
                                       initializer=initializer, trainable=True)

    def encoder(self, feat_mat, dropout=True):
        act_fn = self.act_fn
        y_1 = self.dropout(act_fn(tf.matmul(feat_mat, self.W_1) + self.b_1), mode=dropout)
        y_2 = self.dropout(act_fn(tf.matmul(y_1, self.W_2) + self.b_2), mode=dropout)
        y_3 = self.dropout(act_fn(tf.matmul(y_2, self.W_3) + self.b_3), mode=dropout)
        return y_3

    def decoder(self, repr_vec):
        act_fn = self.act_fn
        y_4 = self.dropout(act_fn(tf.matmul(repr_vec, tf.transpose(self.W_3)) + self.b_4))
        y_5 = self.dropout(act_fn(tf.matmul(y_4, tf.transpose(self.W_2)) + self.b_5))
        y_6 = tf.matmul(y_5, tf.transpose(self.W_1)) + self.b_6
        return y_6

    def model(self):
        self.sess = tf.Session()
        dropout_feat_mat = self.dropout(self.feat_mat, shape=[None, 1, self.feat_size])
        self.X_dropout = tf.reshape(dropout_feat_mat, [-1, self.n_frame * self.feat_size])
        self.X = tf.reshape(self.feat_mat, [-1, self.n_frame * self.feat_size])

        '''encoder'''
        self.repr_dropout = self.encoder(self.X_dropout)
        self.repr = self.encoder(self.X, dropout=False)

        '''decoder'''
        self.decX_dropout = self.decoder(self.repr_dropout)

        self.saver = tf.train.Saver()

    def pretrain(self, trainData):
        n_layer = self.n_layer
        layers = [self.feat_size * self.n_frame] + self.layers
        for i in range(n_layer):
            g = tf.Graph()
            with g.as_default():
                feat_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
                dropout_feat_mat = self.dropout(feat_mat, shape=[None, 1, self.feat_size])
                X = tf.reshape(dropout_feat_mat, [-1, self.n_frame * self.feat_size])
                act_fn = self.act_fn
                initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=7)
                load_variables = {}
                y_list = [X]
                W_list = []
                b_list = []
                for j in range(i + 1):
                    trainable = True if i == j else False
                    n_in, n_out = layers[j], layers[j + 1]
                    W_k = tf.get_variable('W_%d' % (j + 1), shape=[n_in, n_out],
                                          initializer=initializer, trainable=trainable)
                    b_k = tf.get_variable('b_%d' % (j + 1), shape=[1, n_out],
                                          initializer=initializer, trainable=trainable)
                    y_k = self.dropout(act_fn(tf.matmul(y_list[-1], W_k) + b_k))
                    y_list.append(y_k)
                    W_list.append(W_k)
                    b_list.append(b_k)
                    if j != i:
                        load_variables['W_%d' % (j + 1)] = W_list[-1]
                        load_variables['b_%d' % (j + 1)] = b_list[-1]
                    
                for j in range(i + 1):
                    trainable = True if j == 0 else False
                    n_out = layers[i - j]
                    b_k = tf.get_variable('b_%d' % (2*n_layer - (i - j)), shape=[1, n_out],
                                          initializer=initializer, trainable=trainable)
                    y_k = tf.matmul(y_list[-1], tf.transpose(W_list[-(j+1)]) + b_k)
                    if j != i:
                        y_k = self.dropout(act_fn(y_k) )
                    y_list.append(y_k)
                    b_list.append(b_k)
                    if j != 0:
                        load_variables['b_%d' % (2*n_layer - (i - j))] = b_list[-1]
                
                cost = self.square_err(X, y_list[-1])
                optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)
                #g.finalize()

            sess = tf.Session(graph=g)
            with sess:
                sess.run(tf.global_variables_initializer())
                if len(load_variables) != 0:
                    print('Loading pretrained model for pretraining %.2d' % i)
                    saver_load = tf.train.Saver(load_variables)
                    saver_load.restore(sess, "./pretrain_model.ckpt")

                for epoch in range(self.pretrain_n_epoch):
                    total_cost = []
                    for b_step in range(0, len(trainData), self.b_size):
                        x = trainData[b_step: min(len(trainData), b_step + self.b_size)]
                        res = sess.run([cost, optimizer], feed_dict={feat_mat: x})
                        total_cost.append(res[0])
                    total_cost = np.mean(total_cost)
                    print('Pretrain%.2d_epoch%.2d Total_cost %.4f' % (i, epoch, total_cost))

                saver_save = tf.train.Saver()
                saver_save.save(sess, "./pretrain_model.ckpt")

        return

    def load_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, "./speech_autoencoder_dropout.ckpt")
    
    def save_model(self):
        self.saver.save(self.sess, "./speech_autoencoder_dropout.ckpt")
            
    def square_err(self, X, Y):
        return tf.reduce_mean(tf.reduce_sum((X - Y)**2, axis=1))

    def calculateCost(self):
        self.SE = self.square_err(self.X_dropout, self.decX_dropout)
        self.cost = self.SE
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        if self.load == True:
            print('Loading trained model')
            self.load_model()

    def __call__(self, X):
        return self.sess.run(self.cost, feed_dict={self.feat_mat: X})

    def extract(self, X):
        return self.sess.run(self.repr, feed_dict={self.feat_mat: X})
        
    def training(self, n_epoch, trainData, devData, repack_data=None, pretrain=False, load_pretrain=False):
        ''' Pretrain layer'''
        if pretrain:
            self.pretrain(trainData)

        ''' Training start'''
        self.sess.run(tf.global_variables_initializer())
        if load_pretrain:
            print('Loading pretrained model')
            self.saver.restore(self.sess, "./pretrain_model.ckpt")
        n = len(trainData)
        lowest_cost = float("inf")
        for epoch in range(n_epoch):
            total_cost = []
            for i in range(0, n, self.b_size):
                x = trainData[i: min(n, i + self.b_size)]
                res = self.sess.run([self.cost, self.repr, self.optimizer], 
                                    feed_dict={self.feat_mat: x})
                total_cost.append(res[0])
            dev_cost = test(devData, self)
            print('Epoch %.2d\t Total_cost %.4f\t dev_cost %.4f' % \
                  (epoch, np.mean(total_cost), dev_cost))
            if dev_cost < lowest_cost:
                lowest_cose = dev_cost
                self.save_model()
            if repack_data != None:
                repack_data()

            

