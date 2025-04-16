import pandas as pd
#data = pd.read_csv(r"D:\my_projects\CREDIT\data_set\fraudTrain.csv\fraudTrain.csv")
#print(data.head(6))

class ReadData:
    def __init__(self , path):
        self.path = path 
        self.get()

    def get(self):
        data = pd.read_csv(self.path , header = None)
        return data
       
      