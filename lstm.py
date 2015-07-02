import numpy as np
import theano
import theano.tensor as T
from process import process
from ipdb import set_trace as pause

# This object contains the global network
def int_to_label(y, n_classes):

    result = np.zeros((y.shape[0], n_classes))
    for idx,i in enumerate(y):
        result[idx][int(i)] = 1
    return result

class LSTM(object):

    def __init__(self,n_in,n_layers,n_hidden,n_classes):
        print "LSTM"

        # Initialize all the layers and the links between layers
        # First create a single layer example
        # Declare all the attributes :
        self.lr = 0.001
        self.momentum = 0.9
        self.x = T.matrix()
        self.y = T.matrix()
        # Size parameters
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        # Define one layer of LSTM and one layer of softmax
        # Maybe we could add a feedforward layer later

        self.lstm_layer = LSTM_layer(self.x,n_in,n_hidden)
        self.softmax_layer = Softmax_layer(self.lstm_layer.h,n_hidden,n_classes)
        self.params = [self.lstm_layer.W_i,self.lstm_layer.W_c,self.lstm_layer.W_f,self.lstm_layer.W_o,self.lstm_layer.U_i
                       ,self.lstm_layer.U_f,self.lstm_layer.U_c,self.lstm_layer.V_o,self.lstm_layer.bi,self.lstm_layer.bf,
                       self.lstm_layer.bc,self.lstm_layer.bo,self.lstm_layer.h0]


        # cost function we can directly implement it using y_pred ...
        self.cost = T.mean(T.nnet.binary_crossentropy(self.softmax_layer.p_y_given_x, self.y))
        self.grad_cost = T.grad(self.cost, self.params)
        self.cost_fn = theano.function(inputs=[self.x,self.y],outputs=self.cost)
        self.grad_cost_fn =theano.function(inputs=[self.x,self.y],outputs= self.grad_cost)

        # g = T.grad(costs[0], p)
        # g = map(T.as_tensor_variable, g)  # for CudaNdarray
        # self.f_gc = theano.function(inputs, g + costs, on_unused_input='ignore')  # during gradient computation
        # Update law in a form, misses the x*grad part !!!



        # Implement the batching :
    def train(self,gradient_dataset):
        """     
        :param x: Train an example n_steps * m_features
        :param y: n_steps * 1
        :return:
        """
        #gradient = np.zeros(sum(self.sizes), dtype=theano.config.floatX)

        result =[np.zeros(i.get_value().shape) for i in self.params]
        costs = []
        for inputs in gradient_dataset.iterate(update=True):
            # Construct the list of gradient ( one matrix for each param )
            pause()
            result =[(i+j)/ gradient_dataset.number_batches for i, j in zip(result, self.grad_cost_fn(inputs))]
            # Flat the result and add it to the gradient (average over batches )
            #gradient += self.list_to_flat(result[:len(self.p)]) / gradient_dataset.number_batches
            # # We have to check if the gradient is always well definite.
            # Add the cost and the error in the costs

            # Update all the parameters :
        # Update the parameters :
        updates = [i.get_value()-j for i,j in zip(self.params,result)]
        for i in range(len(self.params)):
            self.params[i].set_value(updates[i])

        # Evaluate the cost at this epoch
        mean_cost = 0
        for inputs in  gradient_dataset.iterate(update=True):
            mean_cost+=self.cost_fn(inputs)

        mean_cost/gradient_dataset.number_batches

        print " Cost : "+str(mean_cost)


    def list_to_flat(self, l):
        return np.concatenate([i.flatten() for i in l])

class Softmax_layer(object):

    def __init__(self, xt,n_in, n_classes):
        # Not sure if shared_variable.shape works but it seems so
        W_soft_init = np.asarray(np.random.uniform(size=(n_in,n_classes),
                                          low=-.01, high=.01),
                                          dtype=theano.config.floatX)
        self.W_soft = theano.shared(value=W_soft_init,name='W_soft')
        b_soft_init = np.asarray(np.zeros(n_classes))
        self.b_soft = theano.shared(value = b_soft_init, name='b_soft')
        self.xt = xt
        self.n_classes = n_classes

        def symbolic_softmax(x):
                e = T.exp(x)
                return e / T.sum(e, axis=1)

        self.p_y_given_x = symbolic_softmax(T.dot(self.xt,self.W_soft))
        self.y_pred = T.argmax(self.p_y_given_x, axis=-1)

# This object is a memory cell


class LSTM_layer(object):

    def __init__(self,x,n_input, n_hidden):
        print "Init LSTM"
        # n_candidate = n_output
        # Init weights :

        # Input weights :
        W_i_init = np.asarray(np.random.uniform(size=(n_input, n_hidden),
                                          low=-.01, high=.01),
                                          dtype=theano.config.floatX)

        self.W_i = theano.shared(value=W_i_init,name='W_i')

        # Forget weights  :

        W_f_init = np.asarray(np.random.uniform(size=(n_input, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.W_f = theano.shared(value=W_f_init,name='W_f')

        # Candidate weights :

        W_c_init = np.asarray(np.random.uniform(size=(n_input, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.W_c = theano.shared(value=W_c_init,name='W_c')

        # Output weights
        W_o_init = np.asarray(np.random.uniform(size=(n_input, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.W_o = theano.shared(value=W_o_init,name='W_o')



        # State weights  :
    

        # Input state weights :
        U_i_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                          low=-.01, high=.01),
                                          dtype=theano.config.floatX)

        self.U_i = theano.shared(value=U_i_init,name='U_i')

        # Forget state weights  :

        U_f_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.U_f = theano.shared(value=U_f_init,name='U_f')
        
        
        # Candidates state weights :

        U_c_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.U_c = theano.shared(value=U_c_init,name='U_c')
        
                
        # Output state weights :

        U_o_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.U_o = theano.shared(value=U_o_init,name='U_o')
        
        

        # Output candidate weights
        V_o_init = np.asarray(np.random.uniform(size=(n_hidden, n_hidden),
                                  low=-.01, high=.01),
                                  dtype=theano.config.floatX)
        self.V_o = theano.shared(value=V_o_init,name='V_o')



        # Init hidden state vector :

        h_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.h = theano.shared(value=h_init, name='h')

        # Init biases :

        # forget biases:

        bf_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.bf = theano.shared(value=bf_init, name='bf')

        # input biases :
        bi_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.bi= theano.shared(value=bi_init, name='bi')


        # candidate biases :
        bc_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.bc= theano.shared(value=bc_init, name='bc')
        
        
        # candidate biases :
        bo_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.bo= theano.shared(value=bo_init, name='bo')
        
        ho_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.h0 = theano.shared(value=ho_init,name='h0')

        co_init = np.zeros((1,n_hidden), dtype=theano.config.floatX)
        self.c0 = theano.shared(value=co_init,name='c0')

        self.c = T.matrix()
        
        # xt
        
        # x = [x1,...,xt,...,xn]
        self.x = x


        [self.h,self.c], updates = theano.scan(fn=self.step,sequences=self.x,outputs_info=[self.h0,self.c0])

        # Formulas :

        # Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, Vo

        # it = sigma(Wi*xt + Ui*ht-1 + bi)
        # Ctest = tanh(Wc*xt + Uc*ht-1 + bc)
        # ft = sigma( Wf*xt + Uf*ht-1 + bf )
        # Ct = it* Ctest +ft*Ct-1
        # ot = sigma(Woxt + Uo*ht-1 + VoCt +bo)
        # ht = ot * tanh(Ct)

    def step(self,xt,ht_pre,ct_pre):
        it = T.nnet.sigmoid(T.dot(xt,self.W_i)+T.dot(ht_pre,self.U_i)+self.bi)
        ct_est = T.tanh(T.dot(xt,self.W_c)+T.dot(ht_pre,self.U_c)+self.bc)
        ft = T.nnet.sigmoid(T.dot(xt,self.W_f)+T.dot(ht_pre,self.U_f)+self.bf)
        ct = it*ct_est+ft*ct_pre
        ot = T.nnet.sigmoid(T.dot(xt,self.W_o)+T.dot(ht_pre,self.U_o)+T.dot(ct,self.V_o)+self.bo)
        ht = ot*T.tanh(ct)

        return ht,ct


# Class quite useful for batch the training trough files
class SequenceDataset:
  '''Slices, shuffles and manages a small dataset for the HF optimizer.'''

  def __init__(self, data, batch_size, number_batches, minimum_size=10):
    '''SequenceDataset __init__
  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''

    self.current_batch = 0
    self.number_batches = number_batches
    self.items = []

    for i_sequence in xrange(len(data[0])):
      if batch_size is None:
        self.items.append([data[i][i_sequence] for i in xrange(len(data))])
      else:
        for i_step in xrange(0, len(data[0][i_sequence]) - minimum_size + 1, batch_size):
          self.items.append([data[i][i_sequence][i_step:i_step + batch_size] for i in xrange(len(data))])
    self.shuffle()

  def shuffle(self):
    np.random.shuffle(self.items)

  def iterate(self, update=True):
    # iterate over the dataset returning number_batches [x,y] in an iterable
    for b in xrange(self.number_batches):
      yield self.items[(self.current_batch + b) % len(self.items)]
    # update the current batch ( starting point )
    if update: self.update()

  def update(self):
    if self.current_batch + self.number_batches >= len(self.items):
      self.shuffle()
      self.current_batch = 0
    else:
      self.current_batch += self.number_batches


if __name__ == '__main__':

    trX, trY = process()

    n_in = trX[0].shape[1]
    n_hidden = 20
    n_layers = 1
    n_updates = 10
    n_classes = 7
    # Store the raw classes anyway
    labels = trY
    # Compute probabilistic labelling :
    trY = [int_to_label(i,n_classes) for i in trY]
    gradient_dataset = SequenceDataset([trX, trY], batch_size=None,
                                       number_batches=len(trX))


    model = LSTM(n_in,n_layers,n_hidden,n_classes)


    for i in range(n_updates):
        model.train(gradient_dataset)





