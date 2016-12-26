"""
Minimal character-level Vanilla RNN model. Written by Andrej Karpathy (@karpathy)
BSD License
"""
import numpy as np

# data I/O
#data = open('input.txt', 'r').read() # should be simple plain text file
data = open('fox.txt', 'r').read() # should be simple plain text file
data = data.rstrip('\n'); # GE
#chars = list(set(data)) # orig
chars = ['b','r','o','w','n']
data_size, vocab_size = len(data), len(chars)
print 'data has %d characters, %d unique.' % (data_size, vocab_size)
char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }
print "chars= ", chars

# hyperparameters
hidden_size = 1 # size of hidden layer of neurons # orig 100
seq_length = 2 # number of steps to unroll the RNN for # orig 25
learning_rate = 1e-1 # orig .1

# model parameters
rms = .01 # orig .01  (.1 and .01 work)
#np.random.seed(0)  # GE
Wxh = np.random.randn(hidden_size, vocab_size)*rms # input to hidden
Whh = np.random.randn(hidden_size, hidden_size)*rms # hidden to hidden
Why = np.random.randn(vocab_size, hidden_size)*rms # hidden to output
print "Wxh shape: ", Wxh.shape
print "Whh shape: ", Whh.shape
print "Why shape: ", Why.shape

# BEGIN TEMPORARY FOR DEBUGGING
for i in range(hidden_size):
 for j in range(vocab_size):
  Wxh[i,j] = .3 / (i+j+1)

for i in range(vocab_size):
 for j in range(hidden_size):
  Why[i,j] = .3 / (i+j+1)
# END TEMPORARY FOR DEBUGGING

Whh = Whh * 0. + .3
#Whh *= 0.
#Why *= 0.
Wxh *= 0.

bh = np.zeros((hidden_size, 1)) # hidden bias # orig
by = np.zeros((vocab_size, 1)) # output bias # orig
#bh = np.ones((hidden_size, 1)) # hidden bias
#by = np.ones((vocab_size, 1)) # output bias

#----------------------------------------------------------------------
# using formulas I think are correct (GE)
def lossFunGE(inputs, targets, hprev):
  # only valid for seq_len=1
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  print "prev state: hs[-1]= ", hs[-1]

  loss = 0
  for t in xrange(len(inputs)):
    print "--"
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state # orig
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)

    print "h layer input: ", np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh
    print "y layer input: ", np.dot(Why, hs[t]) + by # 
    print "h layer output (hs): ", hs[t]
    print "y layer output: ", ys[t]
    print "ps layer output: ", ps[t]
    print "loss= ", loss

  # backward pass: compute gradients going backwards
  print "---"
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    print "--"
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here, dL/dys
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    print "Wxh = ", Wxh
    print "Why = ", Why
    print "Whh = ", Whh
    print "(Cross entropy gradient:) dy= ", dy

    # this is correct
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    print "dhnext= ", dhnext
    print "dy= ", dy
    print "dh= np.dot(Why.t,dy)+dhnext= ", dh 
    print "1-hs**2= ", 1-hs[t]*hs[t]

    dWxh += np.dot(dhraw, xs[t].T)
    print "hs[t-1]= ", hs[t-1]
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
    print "Weight Deltas"
    print "dWxh= ", dWxh
    print "dWhy= ", dWhy
    print "dWhh= ", dWhh

    print "Bias Deltas"
    print "dbh=(1-hs**2)*dh ", dbh
    print "dby=dy= ", dby


  print "-----"
  #works without clipping
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  print "#################################################"
  #quit() ##################
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]  # stateful=true
  #return loss, dWxh, dWhh, dWhy, dbh, dby, [[0.]] # stateful=false

#----------------------------------------------------------------------
def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  #print "Wxh= ", Wxh
  #print "Whh= ", Whh
  #print "Why= ", Why
  #print "bh= ", bh
  #print "by= ", by
  # forward pass
  for t in xrange(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    print "hs[t-1]= ", hs[t-1]
    print "h layer input: ", np.dot(Wxh, xs[t]) + bh + np.dot(Whh, hs[t-1])
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state # orig
    print "h layer output: ", hs[t]
    print "y layer input: ", np.dot(Why, hs[t]) + by # 
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    print "y layer output: ", ys[t]
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    print "ps layer output: ", ps[t]
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
    #print "loss +="

  #quit()

  # backward pass: compute gradients going backwards
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(xrange(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
    dWhy += np.dot(dy, hs[t].T)  # dy correct (dL/dy), hs correct (fwd pass)
    dby += dy
    print "(Cross entropy gradient:) dy= ", dy
    print "dby=dy= ", dby

    dh = np.dot(Why.T, dy) + dhnext # backprop into h  (dh must tbe WRONG)
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    print "dh= np.dot(Why.t,dy)= ", dh   # WRONG in one of the codes
    print "1-hs**2= ", 1-hs[t]*hs[t]
    print "dbh=(1-hs**2)*dh ", dbh

    dWxh += np.dot(dhraw, xs[t].T) # xs correct, dhraw wrong
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  #works without clipping
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  #quit() ##################
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

def sample(h, seed_ix, n):
  """ 
  sample a sequence of integers from the model 
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in xrange(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

n, p = 0, 0
mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
mbh, mby = np.zeros_like(bh), np.zeros_like(by) # memory variables for Adagrad
smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0
nb_epochs = 0

# make sure the random numbers remain the same

while True:
  # prepare inputs (we're sweeping from left to right in steps seq_length long)
  if p+seq_length+1 >= len(data) or n == 0: 
    hprev = np.zeros((hidden_size,1)) # reset RNN memory
    p = 0 # go from start of data
    nb_epochs += 1
  inputs = [char_to_ix[ch] for ch in data[p:p+seq_length]]
  targets = [char_to_ix[ch] for ch in data[p+1:p+seq_length+1]]

  # sample from the model now and then
  #if n % 100 == 0:  # orig
  if n % 10 == 0:
    sample_ix = sample(hprev, inputs[0], 200)
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    print 'nb_epochs %d, iter %d, ----\n %s \n----' % (nb_epochs, n, txt )

  # forward seq_length characters through the net and fetch gradient
  print "ENTER LOSS FUN GE ---- n =", n, " ------------"
  loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFunGE(inputs, targets, hprev)
  smooth_loss = smooth_loss * 0.999 + loss * 0.001

  if (n == 5): quit()

  if n % 100 == 0: print 'iter %d, smooth_loss: %f' % (n, smooth_loss) # print progress
  
  # perform parameter update with Adagrad
  for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], 
                                [dWxh, dWhh, dWhy, dbh, dby], 
                                [mWxh, mWhh, mWhy, mbh, mby]):
    mem += dparam * dparam
    #print "mem= ", mem

    disable_backprop = True
    disable_backprop = False

    if disable_backprop == False:
        #param += -learning_rate * dparam / np.sqrt(mem + 1e-8) # adagrad update
        param += -learning_rate * dparam  # SGD

  p += seq_length # move data pointer
  n += 1 # iteration counter 
