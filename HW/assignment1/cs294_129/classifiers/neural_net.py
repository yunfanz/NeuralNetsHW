import numpy as np
import matplotlib.pyplot as plt


class TwoLayerNet(object):
  """
  A two-layer fully-connected neural network. The net has an input dimension of
  N, a hidden layer dimension of H, and performs classification over C classes.
  We train the network with a softmax loss function and L2 regularization on the
  weight matrices. The network uses a ReLU nonlinearity after the first fully
  connected layer.

  In other words, the network has the following architecture:

  input - fully connected layer - ReLU - fully connected layer - softmax

  The outputs of the second fully-connected layer are the scores for each class.
  """

  def __init__(self, input_size, hidden_size, output_size, std=1e-4):
    """
    Initialize the model. Weights are initialized to small random values and
    biases are initialized to zero. Weights and biases are stored in the
    variable self.params, which is a dictionary with the following keys:

    W1: First layer weights; has shape (D, H)
    b1: First layer biases; has shape (H,)
    W2: Second layer weights; has shape (H, C)
    b2: Second layer biases; has shape (C,)

    Inputs:
    - input_size: The dimension D of the input data.
    - hidden_size: The number of neurons H in the hidden layer.
    - output_size: The number of classes C.
    """
    self.params = {}
    self.params['W1'] = std * np.random.randn(input_size, hidden_size)
    self.params['b1'] = np.zeros(hidden_size)
    self.params['W2'] = std * np.random.randn(hidden_size, output_size)
    self.params['b2'] = np.zeros(output_size)

  def loss(self, X, y=None, reg=0.0):
    """
    Compute the loss and gradients for a two layer fully connected neural
    network.

    Inputs:
    - X: Input data of shape (N, D). Each X[i] is a training sample.
    - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
      an integer in the range 0 <= y[i] < C. This parameter is optional; if it
      is not passed then we only return scores, and if it is passed then we
      instead return the loss and gradients.
    - reg: Regularization strength.

    Returns:
    If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
    the score for class c on input X[i].

    If y is not None, instead return a tuple of:
    - loss: Loss (data loss and regularization loss) for this batch of training
      samples.
    - grads: Dictionary mapping parameter names to gradients of those parameters
      with respect to the loss function; has the same keys as self.params.
    """
    # Unpack variables from the params dictionary
    W1, b1 = self.params['W1'], self.params['b1']  # W1 is dim by num_hid
    W2, b2 = self.params['W2'], self.params['b2']  # W2 is num_hid by num_class
    N, D = X.shape

    # Compute the forward pass
    scores = None
    #############################################################################
    # TODO: Perform the forward pass, computing the class scores for the input. #
    # Store the result in the scores variable, which should be an array of      #
    # shape (N, C).                                                             #
    #############################################################################
    S1 = np.dot(X,W1)+b1 #num_train (N) by num_hid
    X2 = np.maximum(S1,0)     #ReLU function
    scores = np.dot(X2,W2)+b2 #numtrain by numclass
    
    # If the targets are not given then jump out, we're done
    if y is None:
      return scores

    # Compute the loss
    loss = None
    #############################################################################
    # TODO: Finish the forward pass, and compute the loss. This should include  #
    # both the data loss and L2 regularization for W1 and W2. Store the result  #
    # in the variable loss, which should be a scalar. Use the Softmax           #
    # classifier loss. So that your results match ours, multiply the            #
    # regularization loss by 0.5                                                #
    #############################################################################
################## adding bs to the Ws ###################
    W1 = np.vstack((W1,b1)); W2 = np.vstack((W2,b2))
    X = np.hstack((X,np.ones((X.shape[0],1)))); X2 = np.hstack((X2,np.ones((X2.shape[0],1))))

    Is = np.arange(N)
    S_correct = scores[Is,y[Is]]   #num_train by 1
    ex_summed = np.sum(np.exp(scores), axis=1) #num_train by 1
    prob = np.exp(S_correct)/ex_summed
    loss = - np.mean(np.log(prob)) + 0.5*reg*(np.sum(W1*W1)+np.sum(W2*W2))

    # Backward pass: compute gradients
    grads = {}
    #############################################################################
    # TODO: Compute the backward pass, computing the derivatives of the weights #
    # and biases. Store the results in the grads dictionary. For example,       #
    # grads['W1'] should store the gradient on W1, and be a matrix of same size #
    #############################################################################
    dW1, dW2 = np.zeros_like(W1),np.zeros_like(W2)
    num_train = N; num_class = max(y)+1; num_hid = S1.shape[1]

    #layer 2
    #X2 is num_train by num_hid # W2 is num_hid by num_class
    I_to_y = np.zeros((num_train,num_class))
    Is = np.arange(num_train)
    I_to_y[Is,y[Is]] = 1
    dL_dS = (-I_to_y + np.exp(scores)/ex_summed.reshape(ex_summed.shape[0],1))/num_train
    dW2 = np.dot(X2.T,dL_dS) 
    dW2 += reg*W2

    #layer 1
    #num_train (N) by num_hid
    #W1 is dim by num_hid
    dX2 = np.dot(dL_dS,W2.T)
    mask = np.zeros((num_train,num_hid))
    cover = np.where(S1>0)
    mask[cover[0],cover[1]] = 1
    dX2 = (dX2.T[:-1]).T*mask    #stripped the b2 column here because it doesn't affect layer 1
    #print dW1.shape, X.shape, mask.shape
    dW1 += np.dot(X.T,dX2) 
    dW1 += reg*W1


    grads['W1'] = dW1[:-1]
    grads['W2'] = dW2[:-1]
    grads['b1'] = dW1[-1]
    grads['b2'] = dW2[-1]

    return loss, grads

  def train(self, X, y, X_val, y_val,
            learning_rate=1e-3, learning_rate_decay=0.95,
            reg=1e-5, num_hiders=100,
            batch_size=200, verbose=False):
    """
    Train this neural network using stochastic gradient descent.

    Inputs:
    - X: A numpy array of shape (N, D) giving training data.
    - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
      X[i] has label c, where 0 <= c < C.
    - X_val: A numpy array of shape (N_val, D) giving validation data.
    - y_val: A numpy array of shape (N_val,) giving validation labels.
    - learning_rate: Scalar giving learning rate for optimization.
    - learning_rate_decay: Scalar giving factor used to decay the learning rate
      after each epoch.
    - reg: Scalar giving regularization strength.
    - num_hiders: Number of steps to take when optimizing.
    - batch_size: Number of training examples to use per step.
    - verbose: boolean; if true print progress during optimization.
    """
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train / batch_size, 1)

    # Use SGD to optimize the parameters in self.model
    loss_history = []
    train_acc_history = []
    val_acc_history = []

    for it in xrange(num_hiders):

      #########################################################################
      # TODO: Create a random minibatch of training data and labels, storing  #
      # them in X_batch and y_batch respectively.                             #
      #########################################################################
      selection = np.random.choice(np.arange(num_train),size=batch_size)
      X_batch = X[selection]#.T
      y_batch = y[selection]

      # Compute loss and gradients using the current minibatch
      loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
      loss_history.append(loss)

      #########################################################################
      # TODO: Use the gradients in the grads dictionary to update the         #
      # parameters of the network (stored in the dictionary self.params)      #
      # using stochastic gradient descent. You'll need to use the gradients   #
      # stored in the grads dictionary defined above.                         #
      #########################################################################
      for key in self.params:
        self.params[key] += -grads[key]*learning_rate

      if verbose and it % 100 == 0:
        print 'iteration %d / %d: loss %f' % (it, num_hiders, loss)

      # Every epoch, check train and val accuracy and decay learning rate.
      if it % iterations_per_epoch == 0:
        # Check accuracy
        train_acc = (self.predict(X_batch) == y_batch).mean()
        val_acc = (self.predict(X_val) == y_val).mean()
        train_acc_history.append(train_acc)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay

    return {
      'loss_history': loss_history,
      'train_acc_history': train_acc_history,
      'val_acc_history': val_acc_history,
    }

  def predict(self, X):
    """
    Use the trained weights of this two-layer network to predict labels for
    data points. For each data point we predict scores for each of the C
    classes, and assign each data point to the class with the highest score.

    Inputs:
    - X: A numpy array of shape (N, D) giving N D-dimensional data points to
      classify.

    Returns:
    - y_pred: A numpy array of shape (N,) giving predicted labels for each of
      the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
      to have class c, where 0 <= c < C.
    """
    y_pred = None

    ###########################################################################
    # TODO: Implement this function; it should be VERY simple!                #
    ###########################################################################
    W1, b1 = self.params['W1'], self.params['b1']  # W1 is dim by num_hid
    W2, b2 = self.params['W2'], self.params['b2']  # W2 is num_hid by num_class

    S1 = np.dot(X,W1)+b1 #num_train (N) by num_hid
    X2 = np.maximum(S1,0)     #ReLU function
    scores = np.dot(X2,W2)+b2 #numtrain by numclass

    y_pred = np.argmax(scores, axis=1)

    return y_pred


