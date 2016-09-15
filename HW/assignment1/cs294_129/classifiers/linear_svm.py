import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dw = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        dw[:,j] += X[i]
        dw[:,y[i]] -= X[i] 

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dw /= num_train

  # Add regularization to the loss.
  loss += 0.5 * reg * np.sum(W * W)
  dw += reg * W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather than first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dw


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  # loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_class = W.shape[1]
  S = X.dot(W) #num_train by num_class
  Is = np.arange(num_train)
  S_correct = S[Is,y[Is]].reshape(S.shape[0],1)*np.ones(S.shape[1])
  #print S.shape, S_correct.shape
  loss = np.sum(np.sum(np.maximum(0,1+S-S_correct),axis=1)-1,axis=0)/num_train+0.5*reg*np.sum(W*W)


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  mask = np.zeros((num_train,num_class))
  cover = np.where(1+S-S_correct>0)
  mask[cover[0],cover[1]] = 1
  num_class_activated = np.unique(cover[0],return_counts=True)
  mask[num_class_activated[0],y[num_class_activated[0]]] -= num_class_activated[1]
  #problem is there are repeated values in cover[0], and only unique copies are iterated over. 
  dW += reg*W 
  dW += X.T.dot(mask)/num_train
  #print X.shape, mask.shape
  return loss, dW
