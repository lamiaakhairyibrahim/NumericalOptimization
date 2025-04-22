from src.code.read_data import ReadData
import pandas as pd
from src.code.Numerical_Optimization import NumericOptim
import matplotlib.pyplot as plt 
from src.code.visualize import visual

rd = ReadData(r"D:\my_projects\NumericalOptimization\NumericOpt\src\dataset\MultipleLR.csv - MultipleLR.csv (1).csv")
data = rd.get()
print(type(rd))
print(data.info())
print(data.shape)
x = data.iloc[:,:3]
y = data.iloc[: , -1]


n = NumericOptim(x , y , lr = 0.00001 , n_itration= 100)
loss , theta = n.Gradient_descent()
v = visual("gradiant descent" , "itration" , "loss" , x= loss)

l , t = n.stochastic_GD_1()
v = visual("stocastic gradiant descent" , "itration" , "loss" , x= l)