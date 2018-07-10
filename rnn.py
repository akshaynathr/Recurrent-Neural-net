import numpy as np 



def rnn_cell_forward(x_t, a_prev, params):
    " Implementaion of a single forward step in RNN cell"

    "x_t --> input data at timestep 't', numpy array of shape(n_x,m)"
    "a_prev --> Hidden state/activation at timestep t-1, numpy array of shape(n_a,m)"
    """params --> contains parameters for the cell as follows
        Wax --> Weight matrix for input X
        Waa --> Weight matric for activation/hidden state "a_t"
        Wya --> Weight matrix for output y
        ba --> bias for a
        by --> bias for y
    """

    """ Function returns:
        a_next --> next hidden state
        yt_pred --> prediction at time step t
        cache --> parameters for backward pass
    """

    # read parameters 
    Wax = params['Wax']
    Wya = params['Wya']
    Waa = params['Waa']
    ba  = params['ba']
    by  = params['by']

    # Compute the next activation

    a_next  = np.tanh(np.dot(Wax,x_t)+np.dot(Wya,a_prev) + ba)
    yt_pred = softmax(np.dot(Wya,a_next) + by)

    cache =(a_next,a_prev, x_t,params)

    return a_next,yt_pred,cache
    