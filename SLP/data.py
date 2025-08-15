import numpy as np

class Data:
    
    def __init__(self):
        #and訓練集
        self.and_x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.and_y_train = np.array([0,0,0,1])

        #or訓練集
        self.or_x_train = np.array([[0,0],[0,1],[1,0],[1,1]])
        self.or_y_train = np.array([0,1,1,1])

        #and 、 or測試集
        self.and_x_test = np.array([[0,1],[0,0],[1,0],[1,1]])
        self.or_x_test = np.array([[0,0],[1,1],[0,1],[1,1]])