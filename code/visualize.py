import matplotlib.pyplot as plt 

class visual:
    def __init__(self , title_1 , titl_x , titl_y , x , y = None):
            self.title_1 = title_1
            self.till_x = titl_x
            self.titl_y = titl_y 
            self.x = x
            self.y = y 
            self.plot()


    def plot(self):
          plt.plot(self.x)
          plt.title(self.title_1)
          plt.xlabel(self.till_x)
          plt.ylabel(self.titl_y)
          plt.show()


        