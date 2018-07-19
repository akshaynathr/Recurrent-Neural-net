# We are making the multiply operation as multiply gate
# Add operation as Add gate.
import numpy as np

class MultiplyGate:
    def forward(self,W,x):
        return np.dot(W,x)

    def backward(self,W,x,dz):
        dW = np.asarray(np.dot(np.transpose(np.asmatrix(dz)), np.asmatrix(x)))      # dW = dz* x
        dx = np.dot(np.transpose(W),dz)                    #dx = dz*W
        return dW,dx



class AddGate:
    def forward(self,x1,x2):
        return x1 +x2

    def backward(self,x2,x1,dz):
        dx1 = dz*np.ones_like(x1)
        dx2 = dz*np.ones_like(x2)
        return dx1 , dx2

    
