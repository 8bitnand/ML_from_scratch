import pandas 

class Car:
    def __init__(self, car_file):
        self.cars = car_file
        df = self.load_csv()
        self.train = self.process_data(df[:-10])
        self.test = self.process_data(df[-10:], train=False)
    
    def load_csv(self):
        
        cardf = pandas.read_csv(self.cars, index_col=0, usecols=[ \
            'seats', 'kms_driven', 'manufacturing_year', 'mileage(kmpl)', 'engine(cc)',\
            'max_power(bhp)', 'torque(Nm)', 'price(in lakhs)'])
        
        return cardf

    
    def process_data(self, df, train=True):
        
        df = df.dropna()
        for col in df.columns:
            df[col] = pandas.to_numeric(df[col], errors='coerce')
            
        df= df.dropna()
        
        if train:
            for col in df.columns:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
            
        return df 
        

c = Car("Linear_regression/Cars.csv")
# print(c.train, c.test)