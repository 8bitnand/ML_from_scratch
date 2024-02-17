import os 
from typing import Tuple 
import pandas 
import numpy
import matplotlib.pyplot as plt 

class Data:
    def __init__(self, path: str):
        self.data = pandas.read_csv(path, names=["sepal_length", "sepal_width",
                                                   "petal_length", "petal_width", "species"])
        self.process_data()
        
    def process_data(self):
        self.gt = self.data["species"]
        self.labels = {label:i for i,label in enumerate(self.gt.unique())}
        self.gt["label_encoading"] = self.data["species"].apply(lambda x: int(self.labels[x]))
        self.x = self.data.drop(["species"], axis=1)
        self.normlalize()


    def normlalize(self):
        for col in self.x.columns:            
            self.x[col] = (self.x[col] - self.x[col].min()) / (self.x[col].max() - self.x[col].min())

           
class KMeansClustring:
    def __init__(self, k) -> None:
        self.k = k
    def distance_d(self, centroids, points):
        
        assert points.shape[-1] == centroids.shape[-1], \
            "centroid and inputs should have same dimensions"
        n, d = points.shape
        dis = numpy.sum((points.reshape((n, 1, d)) - centroids)**2, axis=2)
        dis = numpy.power(dis, 1/d)
        
        return numpy.argmin(dis, axis=1)
        
    def cal_new_centroid(self, dis, points):
        
        for c in range(self.k):
            index = numpy.where(dis==c)
            self.centroids[c] = numpy.mean(points[index], axis=0)
    
    def cal_loss(self, labels, gt):
        
        # TODO Needs to calculate multi class loss 
        return numpy.mean((labels-gt)**2)
        
    def fit(self, x, gt,  epoch=10):
        
        x = x.to_numpy()
        gt = gt.to_numpy()
        labels = None 
        self.centroids = numpy.random.rand(self.k, x.shape[-1])
        for _ in range(epoch):
            labels = self.distance_d(self.centroids, x)
            self.cal_new_centroid(labels, x)
            # print(self.cal_loss(labels, gt))
            
        return labels
       
    

data = Data("K_means_clustering/iris.csv")
kmeans = KMeansClustring(k=4)
labels = kmeans.fit(data.x, data.gt["label_encoading"], epoch=30)
data.data["class"] = labels
print(kmeans.centroids)
print(data.data)
