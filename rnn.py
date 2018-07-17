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



def rnn_forward(x,params):
    """
        1) Create a vector of zeros to store all hidden states.
        2) Initialize the "next" hidden state as a0 (iniital state)
        3) Start looping over each time step, with incremental index 't'
            a)Update the next hidden state and cache by running the rnn_cell_forward
            b)Store the next hidden state in a(t'th position)
            c)Store the prediction in y
            d)Add the cache to list of caches
        4) return a,y,caches

    """
    #Initialize caches
    caches =[]

    n_x,m,Tx = x.shape
    n_y,n_a = params["Wya"].shape

    #initialize y and a with 0
    a= np.zeros((n_a,m,T_x))
    y_pred = np.zeros((n_y,m,T_x))

    # Initialize a_next 
    a_next = a0

    for t in range(T_x):
        a_next,yt_pred,cache = rnn_cell_forward(x[:,:,t],a_next,params)
        caches.append(cache)
        a[:,:,t] = a_next
        y_pred[:,:,t] = yt_pred
    
    caches =(caches,x)
    

    return a,y_pred,caches



def rnn_cell_backward(da_next,cache):
    """ Implementation of backward pass in 
    RNN-cell for single time step 

    da_next--> Gradient of loss wrt next hidden state
    cache  --> contains useful values

    Returns:
    gradients -- <type> python dict
                    dx -- Gradients of input data of shape (n_x,m)
                    da_prev -- Gradients of previous hidden state, of shape (n_a,m)
                    dWax -- Gradients of input-to-hidden weights, of shape (n_a,n_x)
                    dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a,n_a)
                    dba  -- Gradients of bias vector, of shape (n_a,1)
    """

