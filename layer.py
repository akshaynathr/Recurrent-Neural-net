from activations import Tanh
from gate import AddGate, MultiplyGate

mulGate = MultiplyGate()
addGate = AddGate()
activation = Tanh()



class RNNLayer:
    def forward(self,Waa,a_prev,Wax,x):
        """Implements forward pass for the RNN layer.
        Equations:
            a<t> = g(Waa.a<t-1> + Wax.X + ba)
            S<t> = g(W.S<t-1> + U.X + ba)
        """
        self.mulWax_X = mulGate.forward(Wax,x)
        self.mulWaa_a = mulGate.forward(Waa,a_prev)

        self.sum = addGate.forward(self.mulWax_x,self.mulWaa_a)
        self.a_t = activation.forward(self.sum)

        


    
  
