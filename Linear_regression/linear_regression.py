import numpy as np 
from typing import List 
import pandas
from data import Car

class LinearRegression:
    def __init__(self, df: pandas.DataFrame) -> None:
         
        self.y = df["price(in lakhs)"]
        df = df.drop(["price(in lakhs)"], axis=1)
        self.x = df.to_numpy(dtype=np.float32)
        self.num_samples, self.num_features = self.x.shape
        self.weights = np.random.rand(self.num_features)
        self.lr = 0.0001
        self.epoches = 500
    
    def predict(self, row: pandas.DataFrame) -> float:
        
        row = row.to_numpy()
        return np.dot(row, self.weights)    
    
    def train(self): 
        
        for i in range(self.epoches):
            
            y_pred = np.dot(self.weights, self.x.T)
            loss = sum(y_pred - self.y)
            dw = (1/self.num_samples)*sum(self.weights)*(loss)
            self.weights =  self.weights - self.lr*dw    

data = Car("Linear_regression/Cars.csv")
l = LinearRegression(data.train)
l.train()
y = data.test["price(in lakhs)"].tolist()
test = data.test.drop(["price(in lakhs)"], axis=1)
print(l.predict(test), y)
        
     
    
        
        
    