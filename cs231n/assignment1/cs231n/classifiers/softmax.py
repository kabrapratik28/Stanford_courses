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
  f = X.dot(W)                                 # N x C dimensional
  maximum = np.array([np.amax(f, axis=1)]).T
  f = f - maximum
  exp_f = np.exp(f)
  sum_f = np.sum(exp_f,axis=1)                 # N x 1 dimensional
  expected = exp_f[np.arange(num_train),y]     # N x 1 dimensional
  y_cal = expected / sum_f               
  loss = -np.log(y_cal)
  loss = np.sum(loss) / float(num_train)
  
  for each_train in range(num_train):
    X_instance = np.array([X[each_train]]).T
    class_predications = exp_f[each_train] / sum_f[each_train] # 1 x C dimensional
    dW = dW + np.dot(X_instance, np.array([class_predications]))
    dW[:,y[each_train]] = dW[:,y[each_train]] - X[each_train]
  
  loss = loss + 0.5 * reg * np.sum(W * W)
  dW = dW / float(num_train)
  dW = dW + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  f = X.dot(W)                                 # N x C dimensional
  maximum = np.array([np.amax(f, axis=1)]).T
  f = f - maximum
  exp_f = np.exp(f)
  sum_f = np.sum(exp_f,axis=1)                 # N x 1 dimensional
  expected = exp_f[np.arange(num_train),y]     # N x 1 dimensional
  y_cal = expected / sum_f               
  loss = -np.log(y_cal)
  loss = np.sum(loss) / float(num_train)
  
  class_predications = exp_f/np.array([sum_f]).T
  dW = X.T.dot(class_predications)
  subtract_part = np.zeros(f.shape)                 # N x C dimensional
  subtract_part[np.arange(num_train),y] = 1
  dW = dW - X.T.dot(subtract_part)
  
  loss = loss + 0.5 * reg * np.sum(W * W)
  dW = dW / float(num_train)
  dW = dW + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

