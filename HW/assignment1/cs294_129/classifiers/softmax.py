import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss = 0.
  for i in xrange(num_train):
    scores = X[i].dot(W)  #1 by num_class
    scores -= max(scores)   #bias to prevent instability
    sumscores = np.sum(np.exp(scores))
    prob = np.exp(scores[y[i]])/sumscores
    loss -= np.log(prob)
    dW[:,y[i]] += -X[i]
    for j in xrange(num_class): 
      dW[:,j] -= -1./sumscores*np.exp(scores[j])*X[i]
  loss /= num_train
  dW /= num_train

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  num_train = X.shape[0]
  num_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  S = X.dot(W) #num_train by num_class
  S -= np.amax(S,axis=1).reshape(S.shape[0],1)
  Is = np.arange(num_train)
  S_correct = S[Is,y[Is]]   #num_train by 1
  ex_summed = np.sum(np.exp(S), axis=1) #num_train by 1
  prob = np.exp(S_correct)/ex_summed
  loss = - np.mean(np.log(prob))
  I_to_y = np.zeros((num_train,num_class))
  I_to_y[Is,y[Is]] = 1
  dW += -np.dot(X.T,I_to_y)
  #print I_to_y.shape
  dW -= -X.T.dot(np.exp(S)/ex_summed.reshape(ex_summed.shape[0],1))
  dW /= num_train


  return loss, dW

