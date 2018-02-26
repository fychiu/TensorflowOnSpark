import tensorflow as tf
import numpy as np
import os

def test(testData, classifier):
    cost = 0.
    batchsize = 50
    for data in DataBatch(testData,batchsize):
        correct += classifier(data)
    return cost/testData.shape[0]

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
    
    def __init__(self, n_hidden_layers, feat_size, learning_rate, n_frame, b_size=40):
        self.n_layer_1, self.n_layer_2, self.n_layer_3 = n_hidden_layers
        self.n_layer = len(n_hidden_layers)
        self.layers = n_hidden_layers
        self.n_repr = self.n_layer_3
        self.feat_size = feat_size
        self.learning_rate = learning_rate
        self.n_frame = n_frame
        self.b_size = b_size
        self.pretrain_n_epoch = 10
        self.act_fn = tf.tanh
    
    def initPlaceholders(self):
        self.feat_shape = [None, self.n_frame, self.feat_size]
        self.feat_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
        self.X = tf.reshape(self.feat_mat, [-1, self.n_frame * self.feat_size])
        
    def initParams(self):
        with tf.variable_scope('Params'):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=7)
            shape_1 = [self.feat_size * self.n_frame, self.n_layer_1]
            shape_2 = [self.n_layer_1, self.n_layer_2]
            shape_3 = [self.n_layer_2, self.n_layer_3]
            self.W_1 = tf.get_variable('W_1', shape=shape_1, 
                                       initializer=initializer)
            self.W_2 = tf.get_variable('W_2', shape=shape_2, 
                                       initializer=initializer)
            self.W_3 = tf.get_variable('W_3', shape=shape_3, 
                                       initializer=initializer)
            self.b_1 = tf.get_variable('b_1', shape=[1, self.n_layer_1], 
                                       initializer=initializer)
            self.b_2 = tf.get_variable('b_2', shape=[1, self.n_layer_2], 
                                       initializer=initializer)
            self.b_3 = tf.get_variable('b_3', shape=[1, self.n_layer_3], 
                                       initializer=initializer)
            self.b_4 = tf.get_variable('b_4', shape=[1, self.n_layer_2], 
                                       initializer=initializer)
            self.b_5 = tf.get_variable('b_5', shape=[1, self.n_layer_1], 
                                       initializer=initializer)
            self.b_6 = tf.get_variable('b_6', shape=[1, self.feat_size * self.n_frame], 
                                       initializer=initializer)

    def encoder(self, feat_mat):
        self.y_1 = self.act_fn(tf.matmul(feat_mat, self.W_1) + self.b_1)
        self.y_2 = self.act_fn(tf.matmul(self.y_1, self.W_2) + self.b_2)
        self.y_3 = self.act_fn(tf.matmul(self.y_2, self.W_3) + self.b_3)
        return self.y_3

    def decoder(self, repr_vec):
        self.y_4 = self.act_fn(tf.matmul(repr_vec, tf.transpose(self.W_3)) + self.b_4)
        self.y_5 = self.act_fn(tf.matmul(self.y_4, tf.transpose(self.W_2)) + self.b_5)
        self.y_6 = self.act_fn(tf.matmul(self.y_5, tf.transpose(self.W_1)) + self.b_6)
        return self.y_6

    def model(self, load_pretrain=True):
        self.sess = tf.Session()

        '''encoder'''
        self.repr = self.encoder(self.X)

        '''decoder'''
        self.decX = self.decoder(self.repr)

        self.saver = tf.train.Saver()
        if load_pretrain:
            print('Loading pretrained model')
            self.saver.restore(self.sess, "./pretrain_model.ckpt")

    def pretrain(self, trainData):
        n_layer = self.n_layer
        layers = [self.feat_size * self.n_frame] + self.layers
        for i in range(n_layer):
            g = tf.Graph()
            with g.as_default():
                feat_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
                X = tf.reshape(feat_mat, [-1, self.n_frame * self.feat_size])
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
                    y_k = act_fn(tf.matmul(y_list[-1], W_k) + b_k)
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
                    y_k = act_fn(y_k) if j != i else y_k
                    y_list.append(y_k)
                    b_list.append(b_k)
                    if j != 0:
                        load_variables['b_%d' % (2*n_layer - (i - j))] = b_list[-1]
                
                cost = self.square_err(X, y_list[-1])
                optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
                #g.finalize()

            sess = tf.Session(graph=g)
            with sess:
                sess.run(tf.global_variables_initializer())
                if len(load_variables) != 0:
                    print('Loading pretrained model for pretraining %.2d' % i)
                    saver_load = tf.train.Saver(load_variables)
                    saver_load.restore(sess, "./pretrain_model.ckpt")

                for epoch in range(self.pretrain_n_epoch):
                    total_cost = 0.0
                    for b_step in range(0, len(trainData), self.b_size):
                        x = trainData[b_step: min(len(trainData), b_step + self.b_size)]
                        res = sess.run([cost, optimizer], feed_dict={feat_mat: x})
                        total_cost += res[0]
                    print('Pretrain%.2d_epoch%.2d Total_cost %.4f' % (i, epoch, total_cost))

                saver_save = tf.train.Saver()
                saver_save.save(sess, "./pretrain_model.ckpt")

        return

    def load_model(self, model_saved):
        tf.reset_default_graph()
        self.saver.restore(self.sess, model_saved)
    
    def save_model(self):
        self.saver.save(self.sess, "speech_encoding.ckpt")
            
    def square_err(self, X, Y):
        return tf.reduce_mean(tf.reduce_sum((X - Y)**2, axis=1))

    def calculateCost(self):
        with tf.variable_scope('cost'):
            self.SE = self.square_err(self.X, self.decX)
            self.cost = self.SE
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

    def __call__(self, data, Len):
        return self.sess.run(self.repr, feed_dict={self.feat_mat})
        
    def training(self, n_epoch, trainData, devData, repack_data=None):
        ''' Pretrain layer'''
        self.pretrain(trainData)

        ''' Training start'''
        self.sess.run(tf.global_variables_initializer())
        n = len(trainData)
        for epoch in range(n_epoch):
            total_cost = [0.0]
            for i in range(0, n, self.b_size):
                x = trainData[i: min(n, i + self.b_size)]
                res = self.sess.run([self.cost, self.repr, self.optimizer], 
                                    feed_dict={self.feat_mat: x})
                total_cost[0] += res[0]
            print('Epoch %.2d\t Total_cost %.4f' % (epoch, total_cost[0]))
            if repack_data != None:
                repack_data()
            

