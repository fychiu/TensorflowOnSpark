import numpy as np
from pyspark import SparkContext, SparkConf


n_query = 200
train_data = np.random.uniform(size=(10000, 100))
query_data = np.array(train_data[-n_query:])
assert (train_data[-n_query:] == query_data).all()
k = 2

class pseudo_KNN():
    def __init__(self, k, train_data):
        self.train_data = np.array(train_data)
        self.k = k
        return

    def euclidean_distance_sqr(self, x, y):
        return np.sum((x - y)**2, axis=1)
    
    
    def find(self, x):
        global bc_train, bc_k
        assert type(x) == type(np.zeros((1,))), x
        assert x.size == self.bc_train.value.shape[1]
        distances = self.euclidean_distance_sqr(x, self.bc_train.value).tolist()
        indices = list(range(self.bc_train.value.shape[0]))
        sorted_indices = sorted(indices, key=lambda x: distances[x])[:self.bc_k.value]
        sorted_distances = [distances[x] for x in sorted_indices]
        return [sorted_indices, sorted_distances]
        
    def merge(self, x, y):
        return [x[0] + y[0], x[1] + y[1]]
    
    def query(self, query_data):
        sc = SparkContext()
        self.bc_train = sc.broadcast(self.train_data)
        self.bc_k = sc.broadcast(self.k)
        
        indices, distances = [], []
        index, distance = sc.parallelize(query_data).map(self.find).reduce(self.merge)
        indices += index
        distances += distance
        return indices, distances

KNN = pseudo_KNN(k, train_data)
indices, distances = KNN.query(query_data)
print(sorted(indices, key=lambda x: x), len(indices))
print(sorted(distances, key=lambda x: x), len(distances))
