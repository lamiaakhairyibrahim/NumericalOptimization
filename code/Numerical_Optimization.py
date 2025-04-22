import numpy as np 

class NumericOptim:
    
    def __init__(self , x , y , lr , n_itration ):
        self.x = np.array(x)
        self.y = np.array(y).reshape(-1, 1)
        self.lr = lr 
        self.n_itration = n_itration
        

    def Gradient_descent(self):
        shape_data = self.x.shape
        x0 = np.ones([shape_data[0], 1]) 
        data_x = np.concatenate((x0, self.x), axis=1)
        theta = np.zeros((data_x.shape[1], 1))
        m_m = shape_data[0]
        loss = []

        for j in range(self.n_itration):
            h = data_x @ theta
            error = h - self.y
            j_of_theta_0_1 = (1 / (2 * m_m)) * (error.T @ error)
            loss.append(float(j_of_theta_0_1))
            grad_theta = (1 / m_m) * (data_x.T @ error)
            u = self.lr * grad_theta
            theta = theta - u

        return loss, theta
    

    def stochastic_GD_1(self):
        shape_data = self.x.shape
        x0 = np.ones([shape_data[0], 1]) 
        data_x = np.concatenate((x0, self.x), axis=1)
        theta = np.zeros((data_x.shape[1], 1))
        loss = []

        for epoch in range(self.n_itration):
            total_cost = 0
            for i in range(data_x.shape[0]):
                xi = data_x[i:i+1, :]        # Sample input (1, n)
                yi = self.y[i:i+1, :]        # Corresponding target (1, 1)
                prediction = xi @ theta
                error = prediction - yi
                cost = 0.5 * float(error.T @ error)
                total_cost += cost
                grad = xi.T @ error
                theta = theta - self.lr * grad  # Update immediately per sample
            loss.append(total_cost / data_x.shape[0])  # Average loss per epoch

        return loss, theta


































