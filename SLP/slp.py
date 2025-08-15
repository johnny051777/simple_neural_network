from data import Data
import matplotlib.pyplot as plt
import numpy as np


class SLP:
    def __init__(self):
        """初始化
        
            parameter:
            w權重
            b偏移植
        """
        self.w = None
        self.b = 0
        self.result={
            'epoch':[],
            'acc':[],
            'w':[],
            'b':[]
        }
     
    def model(self,x):
        """定義模型
        
            使用 W*x+bias 更新權重
        """
        return np.dot(self.w,x) + self.b >=0
    
    def fit(self,x,y,epoch,lr):
        """訓練
            parameter: x訓練集, y訓練集, epoch, lr學習率
            W*x+b>=0 : 1
            W*x+b<0 : 0
            
        """
    
        self.w = np.random.uniform(-0.5,0.5,x.shape[1])
        self.b = 0
        self.result={
            'epoch':[],
            'acc':[],
            'w':[],
            'b':[]
        }
        
        for i in range(epoch):
            acc = 0
            for xi,yi in zip(x,y):
                pred = self.model(xi)
                if pred and yi==0:
                    self.w -= lr*xi
                    self.b -= lr
                if not pred and yi==1:
                    self.w += lr*xi
                    self.b += lr
                else:
                    acc +=1
            
            self.result['epoch'].append(i)
            self.result['acc'].append(acc/len(x))
            self.result['w'].append(self.w)
            self.result['b'].append(self.b)
            
            #if acc / len(x) ==1:
            #    break
        
    def train_vision(self):
        plt.plot(self.result['epoch'], self.result['acc'])
        plt.title("Training Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.ylim([0, 1])
        plt.show()
    
    def show_result(self):
        print(self.result)
          
if __name__ == "__main__":
    dataset = Data()
    
    and_model = SLP()
    and_model.fit(dataset.and_x_train , dataset.and_y_train , epoch=100, lr=0.05)
    and_model.train_vision()
    print(and_model.show_result())