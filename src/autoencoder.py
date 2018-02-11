import tensorflow as tf

class Autoencoder(object):
    
    def __init__(self, n_hidden, feat_size, learning_rate, n_frame, D, b_size=40):
        self.n_hidden = n_hidden
        self.feat_size = feat_size
        self.learning_rate = learning_rate
        self.n_frame = n_frame
        self.b_size = b_size
        self.D = self.n_hidden
    
    def initPlaceholders(self):
        self.featVec = tf.placeholder(dtype=tf.float32, 
                           shape=[self.b_size, self.n_frame, self.feat_size])
        self.real_n_frame = tf.placeholder(dtype=tf.int32, 
                           shape=[self.b_size])
        self.y = self.featVec
        
    def initParams(self):
        with tf.variable_scope('Params'):
            initializer = tf.truncated_normal_initializer
            self.W = tf.get_variable('W', 
                                     shape=[self.n_hidden, self.D], 
                                     initializer=initializer)
            #self.b_output = tf.get_variable('b_output', shape=[1, self.n_label],
            #                                initializer=tf.truncated_normal_initializer)
    
    def encoder(self, featVec):
        with tf.variable_scope('encoder'):
            GRU_cell = tf.contrib.rnn.GRUCell(self.n_hidden)
            unstackX = tf.unstack(tf.transpose(featVec, [1, 0, 2]), 
                                               num=self.n_frame)
            GRU_out, GRU_state = tf.contrib.rnn.static_rnn(
                                     GRU_cell, 
                                     unstackX, 
                                     dtype=tf.float32, 
                                     sequence_length=self.real_n_frame)
            
            return GRU_out[-1]

    def decoder(self, reprVec):
        with tf.variable_scope('decoder'):
            GRU_cell = tf.contrib.rnn.GRUCell(self.feat_size)
            tileX = tf.tile(tf.expand_dims(reprVec, 0), [self.n_frame, 1, 1])
            unstackX = tf.unstack(tileX, num=self.n_frame)
            GRU_out, GRU_state = tf.contrib.rnn.static_rnn(
                                     GRU_cell, 
                                     unstackX, 
                                     dtype=tf.float32, 
                                     sequence_length=self.real_n_frame)
            
            return tf.transpose(tf.stack(GRU_out), [1, 0, 2])

    def model(self):
        self.representation = self.encoder(self.featVec)
        self.transform = tf.nn.relu(tf.matmul(self.representation, self.W))
        self.output = self.decoder(self.transform)
    
    def calculateCost(self):
        with tf.variable_scope('cost'):
            self.cost = tf.reduce_mean((self.output - self.y)**2)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        
    def training(self, n_epoch, trainX, trainLen):
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(n_epoch):
                total_cost = 0.0
                for i in range(0, len(trainX), self.b_size):
                    X = trainX[i: i + self.b_size]
                    Len = trainLen[i: i + self.b_size]
                    res = sess.run([self.cost, self.optimizer], 
                                   feed_dict={self.featVec: X, 
                                            self.real_n_frame: Len})
                    total_cost += res[0]
                    print('Train %d\t Cost %.4f\t' % (i, res[0]))
                print('Epoch %d\t Total_cost %.4f\t' % (epoch, total_cost))
                

