# We are making the multiply operation as multiply gate
# Add operation as Add gate.
import numpy as np

class MultiplyGate:
    def forward(self,W,x):
        return np.dot(W,x)

    




class AddGate:
    def forward(self,x1,x2):
        return x1 +x2

    
