import matplotlib.pyplot as plt
import torch.nn.functional as F
# %matplotlib inline
import torch

# Cleaning the data is of utmost importance.
# Make data symmetric with the tensor array dimensions.
    # Fill void with zeros for error handling.
# The best way to approach is to get the uniques, sort the data
# Get the data to adhere to the dimensison of the torch tensor.


words_path = r"C:\Users\ujasv\Downloads\names.txt"
words = open(words_path, 'r').read().splitlines()   # list of words

# collapses all words to a consistent string, then making a set to get uniques alphabets only
set_of_chars = set(''.join(words)) 
# sorted alphabets in a list form
chars = sorted(list(set_of_chars))

# Assigning numbers to each alphabet starting from 1, hencee i+1. Alphabets as keys and numbers as vals
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0     # This is used as a token. to tokenize begining/endings of words
# Flipping keys to values and values to keys.
itos = {i:s for s,i in stoi.items()}

total_uniques = len(list(stoi.keys()))
# Making an empty array for unique alphabets format type int32
N = torch.zeros((total_uniques,total_uniques), dtype=torch.int32)      

# Uniques and their occourances in a 2x2 tensor in this section
for w in words:
  # alphabets of each word in a list form with leading and trailing dot `.`
  chs = ['.'] + list(w) + ['.']     # chs is ['.', 'A', 'm', 'y', '.']
  for ch1, ch2 in zip(chs, chs[1:]):    # first iter will be: ch1 as ['.'] and ch2 ['A']
    bi_char = ch1 + ch2   # for 'jade' it'll probs be either 'ja' or 'ad' or 'de'
    ix1 = stoi[ch1]   # getting index of the associated char-alphabet from stoi
    ix2 = stoi[ch2]
    # This will add a one to the preassigned 0 of torch.zeros in N at 
      # the row and column of the associated number of alphabet
      # Essentially collecting the occourances
    N[ix1, ix2] += 1

# Show the uniiques and their occourances in a presentable manner(optional)
plt.figure(figsize=(total_uniques,total_uniques))
plt.imshow(N, cmap='Blues')
for i in range(total_uniques):
    for j in range(total_uniques):
        chstr = itos[i] + itos[j]   # gets first and seccond char alphabets and stores it here
        
        # initial j,i args are the x and y axis assignments. y axis goes from top to bottom unlike bottom to top
          # increments of it are like (0,0), (1,0), (2,0) ...
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()

# Adding a count of 1 all possibilities. Keeps the depth of 0 since dividing by 0 is fround upon.
  # This is called model smoothing. Basically adding every possibilities to
    # make sure you dont get infinity during calculating log liklihood as loss. 
    # The more you add, the more uniform the model. Less you add the more peak/sharp the model 
    # You're pretty much introducing background noise for empty values.
P = (N+1).float()     
# axis 1 will sums by columns: sums 1st row and all columns, and so on for remaining rows. 
  # This part is Normalizing the values keepinng dims intact.
  # Normalizing is just a probablity distribution that scales down numbers to managable size.
    # total should sum to 1 so no number is more than 1.
  # Earlier version shows no keepdims for vector probabilities
P /= P.sum(axis=1, keepdims=True)   # P here is just the normalized N

# This is a manual seed that helps keep outputs consistent to track them.
g = torch.Generator().manual_seed(2147483647)

# Here we generate names.
for i in range(5):
  out = []  # Generated characters are stuffed in here until it finds period(`.`)
  ix = 0      # This is the row of `N`
  while True:
    p = P[ix]   # call row of the probabilities, essentially P[ix,:]
    # This will output the num_samples specified based on the probility distribution provided.
      # eg. p with (0.2,0.3,0.5) will make this output a tensor with
        # roughly(and randomly) 20% 0'a, 30% of 1's and 50% of 2's. that 0,1 and 2 are indexes from probability distro
    # replacement: same index can be drawn more than once (replacement=True) or not (replacement=False).
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    # Apparently here is where the transition occours from row to the 2D index
    out.append(itos[ix])
    if ix == 0:   # if the derived ix from the probability row is 0 its a `.` and word has ended
      break
  print(''.join(itos[i] for i in out))


# Here we estimate the probaliity of the entire training set or just a made up name.
log_likelihood = 0.0      # accumulating all log likihoods here
n = 0     # keeping count of each bi chars log. essentially total amounts of logs
for w in words:   # Getting the probability of entire dataset.
# for w in ["andrejq"]:      # this is to estimate probability of a name through the model.
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    bi_char = ch1 + ch2   # for 'jade' it'll probs be either 'ja' or 'ad' or 'de'
    ix1 = stoi[ch1]   # bring the index of the 1st character of the bi
    ix2 = stoi[ch2]   # bring the index of the 2nd character of the bi
    prob = P[ix1, ix2]    # bring the probability of the combination
    logprob = torch.log(prob)   # convert the prob to log liklihood
    log_likelihood += logprob
    n += 1
    # You can use this print to check each bi's probability
    #print(f'{ch1}{ch2}: {prob:.4f} {logprob:.4f}')
    # If you get infinity you probably havnt done model smmoothing, recall N+1 operation,
      # It's essentially adding fake values to all posibilities.

print(f'{log_likelihood=}')

nll = -log_likelihood
print(f'{nll=}')
print(f'{nll/n}')

# create the training set of bigrams (x,y) given the first char, predict the next one
xs, ys = [], []   # inputes, targets
for w in words:
  chs = ['.'] + list(w) + ['.']
  for ch1, ch2 in zip(chs, chs[1:]):
    ix1 = stoi[ch1]
    ix2 = stoi[ch2]
    xs.append(ix1)
    ys.append(ix2)

# All indexes of the char.
xs = torch.tensor(xs)   # First char index of the bi set
ys = torch.tensor(ys)   # Second char index of the bi set

# num_classes is the biggest number of the dataset.
# one hot encoded values of the input
xenc = F.one_hot(xs, num_classes=27).float()
W = torch.randn((27, 1))    # The random weights (individual inputs, neurons)


logits = xenc @ W # log-counts  # Matrix multiplication to get dot product
# The exponentiation transforms the values in a nonlinear way rather than just scaling
  # This is nonlinear scaled down version of probability distro we saw in P i.e N
  # Exponentiation enlarges differences between values. larger vals get larger and smaller gets smaller.
  # this probably makes the higher probabilities stand out.
counts = logits.exp()     
# 
probs = counts / counts.sum(1, keepdims=True)

# counts and probs both are a softmax func. 
# Exponentiation then normalization is softmax functions.

# Here instead of having the bi chars we have a block size around it
block_size = 3    # Block size around the char

# *********** VVI ********** #
# rows of X are the batches of inputs and their columns are numbers that point to 
  # their associated char. Essentially indexes using itos(indexes to characters).
# X vals is the indexes of the block chars; Y is the target value right after the block
X,Y = [],[]
for w in words:
    context = [0] * block_size    # Initially 3 0's in list
    for ch in w + '.':    # Appending . at the end of the word
        ix = stoi[ch]     # Bring index of the char
        X.append(context) # Add block indexes to the inputs X
        Y.append(ix)      # Add the value to be predicted after the block, to Y
        context = context[1:] + [ix]    # Change the last index of the block to the next char
X = torch.tensor(X)
Y = torch.tensor(Y)

# squeezing/cramming all chars(27) through this layout of two column space(:,2)
C = torch.randn((27,2))
C[5]    # returns the 5th row.
# Doing same exact thing using one-hot enec vals
total_classes = C.shape[0]      # total classes as num of rows
# Attempting to return index using one hot values. Mind the float() since one is long
# Also only a tensor can be passed as first arguent
one_hot_enc = F.one_hot(torch.tensor(5), num_classes=total_classes).float()
# Getting the index using the one hot encoded values
# This sorcery has to be understood more.
one_hot_enc @ C

# substitute to using one hot is using C[X] because its faster:
  # indexing those from C array gives you (m,n) indexes of C with the column of C
    # as last dimension.
  # Consideing C.shape's rows(all chars) are the total classifications and columns are squeezed versions
  # Every (m,n) of inputs X we get output (o) that is the stripped from random C values. (C.shape[-1])
  # Hence the final output shape of this index calling is (m,n,o)
    # where (o) are the index values of C and (m,n) are just shapes retained from input X
emb_X = C[X]  # Instead of using one_hot_enc we use this for faster processing
# It is essentially replacing the character indexes in X (singular items) with
  # rands of C(2D items) 
# We have a 2D array of batches of inputs(rows) and inputs themselves(columns) in X and
  # and we have random 2D array C that is equivalent to the classifications from X's input characters

# *********CONCLUSION*********
# so emb_X is just X with random values initialized through C that retains the depth of X.

X[17,2] # gives the index of the char
# and
C[X][17,2]  # output is the same as
C[X[17,2]]

# **********VVI************
# Long story short to embed all of the ints from X we do C[X]     @ 19:20 of the youtube video
emb = C[X]    # embeddings are essentially inputs projected on to the random tensor
# emb.shape is (x,3,2) and we need to flatten out the 3,2 to shape (x,6)
emb.view(emb.shape[0],6)    # This operation is the same as .reshape

# Since in our case, each batch is of size 3 but calling it in rand C makes it (3,2)
  # The total input in each batch is really makes it 6. (3 times 2)
inputs,neurons = 6,100    # neurons are arbitrary
W1 = torch.randn((inputs,neurons))
b1 = torch.randn((neurons))

# example of dot product and input*weights+bias(matrix multiplication):
# (1,5) one batch of input with 5 inputs in the batch and
  # (5,1) weight's dot product gives one output,the column, that is one neuron's output.
  # the n_columns of this weights matrix are the number of neurons- i.e 1
  # for (2,5) inputs and weights of (5,3) will result in a (2,3) matrix
  # rows(2) in this case are the outputs and columns(3) are the neurons
  # (:,0) will be first neurons outputs; (:,1) will be second neuron's output and so on
  # whatever the individual inputs are(the columns of the first matrix), 
    # the weights(the rows of second matrix) are reflective of those, hence
    # the columns of first matrix should be equal to the row of second matrix
  # High level overview is matrix X is (batch of inputs, individual inputs per batch)
    # and Y is (weights-per ind input,neurons)
  # bias shape is euqal to number of neurons.

# TANH introduces non-linerity and squaches higher +ve end numbers towards 1
  # and lower -ve end numbers towards towards -1
# the saturated higher and lower end numbers makes backprop gradients small.
# IF YOU DONT USE TANH YOU MIGHT HAVE NUMBERS UPWARDS/DOWNWARDS OF +/-70 AND WHEN
# EXPONENTIATED(THE NEXT STEP) IT WILL THROW INFINITY ON +VE END AND 0 ON -VE END
h = torch.tanh(emb.view(emb.shape[0],6) @ W1 + b1)
# h.shape will (totals of emb.vews, total columns of W1) since this is a dot product.

# final layer:
W2 = torch.randn((100, 27))
b2 = torch.randn((27))

logits = h @ W2 + b2
counts = logits.exp()
prob = counts/counts.sum(1, keepdim=True)   # Normalizing by summing each row.
# prob[0].sum()   # should be equal to 1. infact all rows summed should be equal to 1

# Now pluck out the probabilities from prob based on the target value Y indexes
# Probabilites as compared to ground truth values Y
prob_gtv = prob[torch.arange(prob.shape[0]),Y]
loss = -prob_gtv.log().mean()


# Long story short
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
words_path = r"C:\Users\ujasv\Downloads\names.txt"
words = open(words_path, 'r').read().splitlines()   # list of words
set_of_chars = set(''.join(words)) 
chars = sorted(list(set_of_chars))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

block_size = 3    # Block size around the char
X,Y = [],[]
for w in words:
    context = [0] * block_size    # Initially 3 0's in list
    for ch in w + '.':    # Appending . at the end of the word
        ix = stoi[ch]     # Bring index of the char
        X.append(context) # Add block indexes to the inputs X
        Y.append(ix)      # Add the value to be predicted after the block, to Y
        context = context[1:] + [ix]    # Change the last index of the block to the next char
X = torch.tensor(X)
Y = torch.tensor(Y)


g = torch.Generator().manual_seed(2147483647)
C = torch.randn((27,2),generator=g)
# inputs size is 6 since the block size is 3 and C columns are 2. (3 times 2)
inputs,neurons = 6,100    # neurons are arbitrary
W1 = torch.randn((inputs,neurons),generator=g)
b1 = torch.randn((neurons),generator=g)
W2 = torch.randn((neurons, 27),generator=g)
b2 = torch.randn((27),generator=g)
parameters = [C, W1, b1, W2, b2]

# nelement is the product of the .shape. its the size of the tensor array.
# nelement for shape of (11,3)  is 48
print('total params size:', sum(p.nelement() for p in parameters))

for p in parameters:
   p.requires_grad = True

# Doing lengthy forward/backwarrd pass. Not very efficient. Check minibatch for loop
for _ in range(10):
  # forward pass
  emb = C[X]    # embeddings are essentially inputs
  h = torch.tanh(emb.view(emb.shape[0],inputs) @ W1 + b1)
  logits = h @ W2 + b2
  # counts = logits.exp()
  # prob = counts/counts.sum(1, keepdim=True)   # Normalizing by summing each row.
  # prob_gtv = prob[torch.arange(prob.shape[0]),Y]
  # loss = -prob_gtv.log().mean()

  # Rather than exponentiating logits(counts), normalizing them(prob), 
    # getting the associated probabilities through Y ground truth vals(prob_gtv), and
    # then getting -ve log liklihood as loss
    # we just use the F.cross_entropy instead of all of this lengthy process
  loss = F.cross_entropy(logits,Y)
  print(loss)


  # backward_pass
  for p in parameters:
    p.grad = None
  loss.backward()
  for p in parameters:
    p.data += -0.1 * p.grad   

# Doing this forward backward on a huge dataset is not efficient and takes forever
  # better way is to take tinky chunks of batch(minibatch) and 
  # do forward/back pass on it

# Doing minibatch forward/backward pass instead of whole batch for fast operation
    # and using non-linear learning rate
# Learning rate decay: lowering learning rate when loss starts plateu/becomes unstable.
lre = torch.linspace(-3,0,1000)    # the Learning rate exponent. equidistant nums
lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
# keeping learning rate and loss counter:
lrei,lri,lossi = [],[],[]

# Checking when the losses start to stagnate. This slope when approaches zero
  # is when the loss stagnates, slopes above zero shows increasing losses.
  # slopes close to -ve zero and std closest to zero is where the ideal learning rate exists
  # The data is split to 10 batches each batch containing 100 incremental losses 
a = torch.tensor(lossi).reshape(10,100)
for i in range(a.shape[0]):
    print(torch.diff(a[i]).float().mean().item(),a[i].std())

for i in range(1000):
  # minibatch
  mini_batch_size = 32
  ix = torch.randint(0,X.shape[0],(mini_batch_size,))    # mind the comma ',' for size.
  
  # forward pass
  # The minibatch takes only the 32 rows of X at a time as embeddings
    # so embeddings will be (32,2,3) instead of whole (200K,2,3)
  emb = C[X[ix]]    # embeddings are essentially inputs
  h = torch.tanh(emb.view(emb.shape[0],inputs) @ W1 + b1)
  logits = h @ W2 + b2
  loss = F.cross_entropy(logits,Y[ix])

  # backward_pass
  for p in parameters:
    p.grad = None
  loss.backward()

  # learning rate decay:
  lr = lrs[i]
  for p in parameters:
    p.data += -lr * p.grad
  
  lrei.append(lre[i])
  lri.append(lr)
  lossi.append(loss.item())

# check how learning rate decay affects loss
plt.plot(lri,lossi)
plt.plot(lrei,lossi)

# seems like 0.1 works out the best learning rate

# Evaluating loss for the whole batch after minibatch operation
emb = C[X]    # embeddings are essentially inputs
h = torch.tanh(emb.view(emb.shape[0],inputs) @ W1 + b1)
logits = h @ W2 + b2
# indexing those 32 minibatch ground truth values with ix
loss = F.cross_entropy(logits,Y)
print(loss)

# Checkout MakeMore_OOP.py for a more object oriented approach.
# Lower loss does not necessarily mean better model.
# As the total params(more neurons, C projections) grows it becomes more and
# more capable of over-fitting the training set.
# Meaning your loss can eventually become as low as zero. 
  # This is because the model is memorizing the dataset verbatim
# So if you try sampling from that kind of model you will get results exactly as in training set.
# In adition to that if you derive a loss from a name outside of a dataset 
# you can get a very high loss.

# For this reason the dataset needs to be split into 3 parts
  # with approx size of 70,20,10 %s respectively.
# Training split, dev split(validation split) and finally the test split.


# sampling from the model to extrace the generated names.
g = torch.Generator().manual_seed(2147483647+30)
for _ in range(20):
  out = []
  context = [0] *block_size
  while True:
    emb = C[torch.tensor(context)]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(1,inputs) @ W1 + b1)
    logits = h @ W2 + b2
    prob = F.softmax(logits,dim=1)
    ix = torch.multinomial(prob, num_samples=1).item()
    context = context[1:]+[ix]
    out.append(ix)
    if ix == 0:
        break
    #   loss = F.cross_entropy(logits,Y)
    #   print(loss)
  print(''.join(itos[i] for i in out))

# evaliat