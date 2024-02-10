from typing import Any
import matplotlib.pyplot as plt 

def plot_fig(data, mode="train"):
    
    cm = ["r", "g", "b", "y"]
    
    x = [x[0] for x in data ]
    y = [x[1] for x in data ]
    c = [cm[x[2]] for x in data ]

    plt.scatter(x, y, c=c)
    plt.savefig(f"KNN/{mode}_iris.png")
    
    

class Data:
    def __init__(self, irispath) -> None:
        self.irispath = irispath

    def load_data(self):
        data = []
        with open(self.irispath) as f:
            for line in f:
                f1, _,f2,_, label = line.split(",")
                data.append((float(f1), float(f2), label.strip()))
        
        return data  
        
    def train_test_split(self, data):
        
        label_map = dict()
        for item in data:
            if item[-1] not in label_map:
                label_map[item[-1]] = []
                
            label_map[item[-1]].append(item[:-1])
        
        train_test_data = {"train":[], "test": []}
        label2index = {label: i for i, label in enumerate(label_map.keys())}
        
        for key in label_map:
            for item in label_map[key][:-10]:
                train_test_data["train"].append((*item, label2index[key]))
                
            for item in label_map[key][-10:]:
                train_test_data["test"].append((*item, label2index[key]))
        
        return train_test_data , label2index


class KNN:
    def __init__(self, train_test_data, labels, k) -> None:
        self.train = train_test_data["train"]
        self.test = train_test_data["test"]
        self.labels = labels
        self.k = k 
    
    def distance(self, test_dp):
        
        def Euclid_discance(p1, p2):
            import math 
            return math.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
            
        d = []
        for dp in self.train:

            d.append((Euclid_discance(test_dp, dp), dp[-1]))
       
        return sorted(d, key=lambda x:x[0])
           
    def get_max_occurance(self, distance):
        
        hashmap = {}
        
        for item in distance:
            if item[-1] not in hashmap:
                hashmap[item[-1]] = 0
                
            hashmap[item[-1]] += 1
                
        return max(hashmap)
            
    def predict(self):
        for x in self.test:
            d = self.distance(x) 
            print(x)
            print(self.get_max_occurance(d[:self.k]))
            print(x[-1])
            print()
            

data = Data("KNN/iris.csv")   
train_test_data, labels = data.train_test_split(data.load_data()) 

# plot_fig(train_test_data["train"] )
# plot_fig(train_test_data["test"], "test")

knn = KNN(train_test_data, labels, k=5)
knn.predict()