import tensorflow as tf
import numpy as np

def test(testData, testLen, classifier):
    correct=0.
    batchsize = 50
    n = testData[0].shape[0]
    for data, Len in DataBatch(testData,testLen,batchsize):
        correct += sum(classifier(data, Len))
    return correct/n*100

def DataBatch(testData, testLen, batchsize, shuffle=True):
    testQue, testPos, testNeg = testData
    testQueLen, testPosLen, testNegLen = testLen
    n = testQue.shape[0]
    if shuffle:
        index = np.random.permutation(n)
    else:
        index = np.arange(n)
    for i in range(int(np.ceil(n/batchsize))):
        inds = index[i*batchsize : min(n,(i+1)*batchsize)]
        yield [testQue[inds], testPos[inds], testNeg[inds]], \
              [testQueLen[inds], testPosLen[inds], testNegLen[inds]]

class Autoencoder(object):
    
    def __init__(self, n_hidden, feat_size, learning_rate, n_frame, b_size=40, load=True):
        self.n_repr = n_hidden
        self.feat_size = feat_size
        self.learning_rate = learning_rate
        self.n_frame = n_frame
        self.b_size = b_size
        self.n_tran = self.n_repr
        self.load = load
    
    def dropout(self, X, rate=0.1, shape=None, mode=True):
        return tf.layers.dropout(X, rate=rate, noise_shape=shape, seed=7, training=mode)

    def initPlaceholders(self):
        self.feat_shape = [None, self.n_frame, self.feat_size]
        self.que_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
        self.pos_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
        self.neg_mat = tf.placeholder(dtype=tf.float32, shape=self.feat_shape)
        self.que_n_frame = tf.placeholder(dtype=tf.int32, shape=[None])
        self.pos_n_frame = tf.placeholder(dtype=tf.int32, shape=[None])
        self.neg_n_frame = tf.placeholder(dtype=tf.int32, shape=[None])
        self.n_files = tf.placeholder(dtype=tf.int32)
        
    def initParams(self):
        with tf.variable_scope('Params'):
            initializer = tf.random_uniform_initializer(-1.0, 1.0, seed=7)
            self.W = tf.get_variable('W', 
                                     shape=[self.n_repr, self.n_tran], 
                                     initializer=initializer)
            self.W_decode = tf.get_variable('W_decode', 
                                     shape=[self.feat_size, self.feat_size], 
                                     initializer=initializer)
            self.b_decode = tf.get_variable('b_decode', 
                                     shape=[1, self.feat_size], 
                                     initializer=initializer)
    
    def encoder(self, feat_mat, feat_n_frame):
        unstackX = tf.unstack(tf.transpose(feat_mat, [1, 0, 2]), num=self.n_frame)
        with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE) as scope:
            state = tf.fill([tf.shape(feat_n_frame)[0], self.n_repr], 0.0)
            cell = tf.fill([tf.shape(feat_n_frame)[0], self.n_repr], 0.0)
            rnn_outputs = []
            for tstep, current_input in enumerate(unstackX):
                if tstep > 0:
                    scope.reuse_variables()
                # input, forget, output, gate
                Hi = tf.get_variable('HMi', [self.n_repr, self.n_repr])
                Hf = tf.get_variable('HMf', [self.n_repr, self.n_repr])
                Ho = tf.get_variable('HMo', [self.n_repr, self.n_repr])
                Hg = tf.get_variable('HMg', [self.n_repr, self.n_repr])

                Ii = tf.get_variable('IMi', [self.feat_size, self.n_repr])
                If = tf.get_variable('IMf', [self.feat_size, self.n_repr])
                Io = tf.get_variable('IMo', [self.feat_size, self.n_repr])
                Ig = tf.get_variable('IMg', [self.feat_size, self.n_repr])

                bi = tf.get_variable('bi', [self.n_repr])
                bf = tf.get_variable('bf', [self.n_repr])
                bo = tf.get_variable('bo', [self.n_repr])
                bg = tf.get_variable('bg', [self.n_repr])

                i = tf.nn.sigmoid(tf.matmul(state, Hi) + tf.matmul(current_input, Ii) + bi)
                f = tf.nn.sigmoid(tf.matmul(state, Hf) + tf.matmul(current_input, If) + bf)
                o = tf.nn.sigmoid(tf.matmul(state, Ho) + tf.matmul(current_input, Io) + bo)
                g = tf.nn.tanh(tf.matmul(state, Hg) + tf.matmul(current_input, Ig) + bg)

                cell = tf.multiply(cell, f) + tf.multiply(g, i)
                state = tf.multiply(tf.nn.tanh(cell), o)

                rnn_outputs.append(state)
            rnn_outputs = tf.stack(rnn_outputs)
            batch_steps = tf.range(tf.shape(feat_n_frame)[0])
            indices = tf.stack([feat_n_frame, batch_steps], axis=1)
            self.final_state = tf.gather_nd(rnn_outputs, indices)

        return self.final_state

    def decoder(self, repr_vec, feat_n_frame):
        tileX = tf.tile(tf.expand_dims(repr_vec, 0), [self.n_frame, 1, 1])
        unstackX = tf.unstack(tileX, num=self.n_frame)
        with tf.variable_scope('Decoder', reuse=tf.AUTO_REUSE) as scope:
            state = tf.zeros([tf.shape(feat_n_frame)[0], self.feat_size])
            cell = tf.zeros([tf.shape(feat_n_frame)[0], self.feat_size])
            rnn_outputs = []
            for tstep, current_input in enumerate(unstackX):
                if tstep > 0:
                    scope.reuse_variables()
                # input, forget, output, gate
                Hi = tf.get_variable('HMi', [self.feat_size, self.feat_size])
                Hf = tf.get_variable('HMf', [self.feat_size, self.feat_size])
                Ho = tf.get_variable('HMo', [self.feat_size, self.feat_size])
                Hg = tf.get_variable('HMg', [self.feat_size, self.feat_size])

                Ii = tf.get_variable('IMi', [self.n_repr, self.feat_size])
                If = tf.get_variable('IMf', [self.n_repr, self.feat_size])
                Io = tf.get_variable('IMo', [self.n_repr, self.feat_size])
                Ig = tf.get_variable('IMg', [self.n_repr, self.feat_size])

                bi = tf.get_variable('bi', [self.feat_size])
                bf = tf.get_variable('bf', [self.feat_size])
                bo = tf.get_variable('bo', [self.feat_size])
                bg = tf.get_variable('bg', [self.feat_size])

                i = tf.nn.sigmoid(tf.matmul(state, Hi) + tf.matmul(current_input, Ii) + bi)
                f = tf.nn.sigmoid(tf.matmul(state, Hf) + tf.matmul(current_input, If) + bf)
                o = tf.nn.sigmoid(tf.matmul(state, Ho) + tf.matmul(current_input, Io) + bo)
                g = tf.nn.tanh(tf.matmul(state, Hg) + tf.matmul(current_input, Ig) + bg)

                cell = tf.multiply(cell, f) + tf.multiply(g, i)
                state = tf.multiply(tf.nn.tanh(cell), o)

                rnn_outputs.append(state)
            rnn_outputs = tf.stack(rnn_outputs)
        return tf.transpose(rnn_outputs, [1, 0, 2])

    def norm(self, X):
        ### X should be N x K matrix where N is n_data and K is n_feat
        return tf.norm(X, axis=1, keepdims=True)

    def cos_sim(self, X, Y):
        ### X and Y should be both N x K matrix where N is n_data and K is n_feat
        inner_prod = tf.reduce_sum(tf.multiply(X, Y), axis=1, keepdims=True)
        norm = tf.multiply(self.norm(X), self.norm(Y))
        cos = tf.realdiv(inner_prod, norm)
        return cos 

    def euclidean_dis(self, X, Y):
        return tf.reduce_sum((X - Y)**2, axis=1, keepdims=True)
        

    def output_decode(self, X):
        flatX = tf.convert_to_tensor(tf.reshape(X, [-1, self.feat_size]))
        res = tf.matmul(flatX, self.W_decode) + self.b_decode
        return tf.reshape(res, [-1, self.n_frame, self.feat_size])

    def model(self):
        self.sess = tf.Session()

        '''encoder'''
        self.que_repr = tf.matmul(self.encoder(self.que_mat, self.que_n_frame), self.W)
        self.pos_repr = tf.matmul(self.encoder(self.pos_mat, self.pos_n_frame), self.W)
        self.neg_repr = tf.matmul(self.encoder(self.neg_mat, self.neg_n_frame), self.W)
        '''
        self.feat_mats = tf.concat([self.que_mat, self.pos_mat, self.neg_mat], axis=0)
        self.feat_n_frame = tf.concat([self.que_n_frame, self.pos_n_frame, self.neg_n_frame], axis=0)
        self.reprs = tf.matmul(self.encoder(self.feat_mats, self.feat_n_frame), self.W)
        self.que_repr = self.reprs[:self.n_files]
        self.pos_repr = self.reprs[self.n_files:-self.n_files]
        self.neg_repr = self.reprs[-self.n_files:]
        '''
  
        '''cosSimilarity'''
        self.pos_cos_sim = self.cos_sim(self.que_repr, self.pos_repr)
        self.neg_cos_sim = self.cos_sim(self.que_repr, self.neg_repr)

        '''norm2_distance'''
        self.pos_distance = self.euclidean_dis(self.que_repr, self.pos_repr)
        self.neg_distance = self.euclidean_dis(self.que_repr, self.neg_repr)

        '''decoder'''
        self.pos_dec = self.decoder(self.pos_repr, self.pos_n_frame)
        self.neg_dec = self.decoder(self.neg_repr, self.neg_n_frame)
        self.pos_out = self.output_decode(self.pos_dec)
        self.neg_out = self.output_decode(self.neg_dec)
        '''
        self.feat_decs = self.decoder(self.feat_mats, self.feat_n_frame)
        self.feat_outs = self.output_decode(self.feat_decs)
        self.que_out = self.feat_outs[:self.n_files]
        self.pos_out = self.feat_outs[self.n_files:-self.n_files]
        self.neg_out = self.feat_outs[-self.n_files:]
        '''

        self.saver = tf.train.Saver()
    
    def load_model(self):
        self.sess.run(tf.global_variables_initializer())
        self.saver.restore(self.sess, "./speech_model.ckpt")
    
    def save_model(self):
        self.saver.save(self.sess, "./speech_model.ckpt")
            
    def square_err(self, X, Y):
        return tf.reduce_mean((X - Y)**2)

    def hinge_loss(self, X, Y):
        return tf.reduce_mean(tf.maximum(0.0, 1 - X + Y), name='max_margin')

    def calculateCost(self):
        with tf.variable_scope('cost'):
            #self.get_correct = self.pos_distance < self.neg_distance
            self.get_correct = self.pos_cos_sim > self.neg_cos_sim
            self.SE = self.square_err(self.pos_dec, self.pos_mat) + \
                      self.square_err(self.neg_dec, self.neg_mat)
            self.HL = self.hinge_loss(self.pos_cos_sim, self.neg_cos_sim)
            self.cost = self.HL + 1e-3*self.SE
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        if self.load == True:
            print('Loading trained model')
            self.load_model()

    def __call__(self, data, Len):
        que, pos, neg = data
        que_len, pos_len, neg_len = Len
        return self.sess.run(self.get_correct,
                                   feed_dict={self.que_mat: que, 
                                              self.pos_mat: pos,
                                              self.neg_mat: neg,
                                              self.que_n_frame: que_len,
                                              self.pos_n_frame: pos_len,
                                              self.neg_n_frame: neg_len,
                                              self.n_files: len(que),
                                             })
        
    def extract(self, X, Len):
        return self.sess.run(self.que_repr, feed_dict={self.que_mat: X, 
                                                       self.que_n_frame: Len})
        
    def training(self, n_epoch, trainData, trainLen, devData, devLen, repack_data=None):
        self.sess.run(tf.global_variables_initializer())

        trainQue, trainPos, trainNeg = trainData
        trainQueLen, trainPosLen, trainNegLen = trainLen
        devQue, devPos, devNeg = devData
        devQueLen, devPosLen, devNegLen = devLen

        highest_acc = 70.0
        for epoch in range(n_epoch):
            total_cost = []
            se_cost = []
            acc = 0.0
            count = 0.0
            for i in range(0, len(trainQue), self.b_size):
                que = trainQue[i: i + self.b_size]
                pos = trainPos[i: i + self.b_size]
                neg = trainNeg[i: i + self.b_size]
                que_len = trainQueLen[i: i + self.b_size]
                pos_len = trainPosLen[i: i + self.b_size]
                neg_len = trainNegLen[i: i + self.b_size]
                res = self.sess.run([self.cost,
                                     self.SE, self.pos_cos_sim, self.neg_cos_sim, 
                                     self.get_correct, self.optimizer, 
                                     self.que_repr, self.pos_repr, self.neg_repr], 
                                    feed_dict={self.que_mat: que, 
                                               self.pos_mat: pos,
                                               self.neg_mat: neg,
                                               self.que_n_frame: que_len,
                                               self.pos_n_frame: pos_len,
                                               self.neg_n_frame: neg_len,
                                               self.n_files: len(que),
                                              })
                total_cost.append(res[0])
                se_cost.append(res[1])
                acc += sum(res[4])
                count += self.b_size
            dev_acc = test(devData, devLen, self)
            print('Epoch %.2d\t Total_cost %.4f\t SE_cost %.4f\t acc %.4f\t dev_acc %.4f' % (epoch, np.mean(total_cost), np.mean(se_cost), acc/count, dev_acc))
            if dev_acc > highest_acc:
                highest_acc = dev_acc
                self.save_model()
            if repack_data != None:
                repack_data()
            

