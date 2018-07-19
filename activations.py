import numpy as np 


class Sigmoid:
    def forward(self,val):
        return 1./(1.0+np.exp(-val) ) # SIGMOID FUNCTION

    
    def backward(self,val,diff_val):
        output = self.forward(val)  # output is value of sigma(x)
        return ((1.0 - output)*  output * diff_val  ) # derivative of SIGMOID FUNCTION --> sigma(x)* (1-sigma(x))




class Tanh:
    def forward(self,val):
        return np.tanh(val)

    def backward(self,val,diff_val):
        output = self.forward(val)
        return ((1.0 - np.square(output))* diff_val)

