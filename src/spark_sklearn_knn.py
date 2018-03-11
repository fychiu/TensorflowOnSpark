import numpy as np
import time
from pyspark import SparkContext, SparkConf
from sklearn.neighbors import NearestNeighbors


n_query = 2700
n_feature = 100
train_data = np.random.uniform(size=(2000000, n_feature))
query_data = np.array(train_data[-n_query:])
assert (train_data[-n_query:] == query_data).all()
k = 2


class pseudo_KNN():
    def __init__(self, k, train_data):
        self.train_data = np.array(train_data)
        self.k = k
        return

    def find(self, x):
        assert type(x) == type(np.zeros((1,))), x
        x = np.reshape(x, (-1, n_feature)) 
        distances, indices = self.bc_knn.value.kneighbors(x)
        return [distances.tolist(), indices.tolist()]
        
    def merge(self, x, y):
        return [x[0] + y[0], x[1] + y[1]]
    
    def query(self, query_data, knnalgorithm='ball_tree'):
        sc = SparkContext()
        knn = NearestNeighbors(n_neighbors=self.k, algorithm=knnalgorithm).fit(self.train_data)
        self.bc_knn = sc.broadcast(knn)
        
        indices, distances = [], []
        distance, index = sc.parallelize(query_data).map(self.find).reduce(self.merge)
        indices += index
        distances += distance
        return indices, distances

KNN = pseudo_KNN(k, train_data)
print('Start query')
start = time.time()
indices, distances = KNN.query(query_data)
print(sorted(indices, key=lambda x: x), len(indices))
print(sorted(distances, key=lambda x: x), len(distances))
print('Spend %.4f' % (time.time() - start))
