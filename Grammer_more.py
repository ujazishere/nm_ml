import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from itertools import product
import pickle
import re

# The IRS Split and generation

# ******VVI******   Convert text to ANSI using Saveas to avoid charmap codec error
# file = "English_grammer.txt"
# file = "basic_eng_min2.txt"
file = "basic_eng.txt"
words_path = rf"C:\Users\ujasv\OneDrive\Desktop\{file}"    
words = open(words_path, 'r').read().splitlines()   # lines

# this trims long items to short
trim_upto = 10         # any lines about this amout of words will be trimmed out
words = [i for i in words if len(i.split())<trim_upto]

"""
words_processed = []
for lines in words:
    list_of_words = lines.split(' ')
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] 
    list_of_words += [list_of_words[-1]]
    words_processed+=list_of_words

"""




# This one is a permutation attempt. basic level 2
continium=[]
for lines in words:
    list_of_words = lines.split(' ')
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    for i in range(len(list_of_words)):
        for y in range(len(list_of_words)):
            aa = list_of_words[i:y+1]
            if aa:
                continium.append(''.join(aa))
words_processed = continium
# old code for continium
"""
continium = []
for lines in words:
    list_of_words = lines.split(' ')
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    for i in range(len(list_of_words)):
        continium.append(''.join(list_of_words[:-i]))
        continium.append(list_of_words[i]+''.join(list_of_words[i+1:]))
    break

"""

set_of_chars = set(words_processed)
chars = sorted(list(set_of_chars))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['..'] = 0
itos = {i:s for s,i in stoi.items()}
output_classes = len(list(stoi.keys()))

"""
# words to their number
words_processed = []
for lines in trimmed_words:
    list_of_words = lines.split(' ')
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    words_processed.append([stoi[y] for y in list_of_words])
"""     

class Make_more:
  def __init__(self, mm_block_size,
                      C_proj_clms,
                      neurons,
                      mini_batch_size,
                      output_classes_or_neurons,
                      gen=None,
                      parameters_input=None,
                      ):
    
    self.mm_block_size = mm_block_size
    self.C_proj_clms = C_proj_clms
    self.neurons = neurons
    self.mini_batch_size = mini_batch_size
    self.output_classes_or_neurons = output_classes_or_neurons
    self.gen = gen
    self.parameters_input = parameters_input

    self.C_iinput_size =  self.C_proj_clms * self.mm_block_size
    
    if not self.gen:      # generations here since its clogging processor
       self.gen = 1000
       
    # Not necessary to declare outputs to be used outside of class.
      # Unless using to create bool for bypassing entensive processing
    self.parameters_output = None

  def initial_cats(self, words, og=None):      # call this to declare X and Y
      block_size = self.mm_block_size  # Block size around the char
      X,Y = [],[]

      if og:
        for w in words:
            context = [0] * block_size    # Initially 3 0's in list
            for ch in w + '.':    # Appending . at the end of the word
                ix = stoi[ch]     # Bring index of the char
                X.append(context) # Add block indexes to the inputs X
                Y.append(ix)      # Add the value to be predicted after the block, to Y
                context = context[1:] + [ix]    # Change the last index of the block to the next char
        X = torch.tensor(X)
        Y = torch.tensor(Y)
      else:
        for lines in words:
            context = [0] * block_size    # Initially 3 0's in list
            list_of_words = lines.split(' ')
            list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
            for w in list_of_words+[itos[0]]:    # Appending seperator at the end of the sentence-'..'
                ix = stoi[w]     # Bring index of the char
                X.append(context) # Add block indexes to the inputs X
                Y.append(ix)      # Add the value to be predicted after the block, to Y
                context = context[1:] + [ix]    # Change the last index of the block to the next char
        X = torch.tensor(X)
        Y = torch.tensor(Y)
      self.X = X
      self.Y = Y


  def params(self, ):
    self.unique_items = len(list(stoi.keys()))

    g = torch.Generator().manual_seed(2147483647)
    
    self.C = torch.randn((self.unique_items,self.C_proj_clms),generator=g)
    
    self.W1 = torch.randn((self.C_iinput_size,self.neurons),generator=g)
    self.b1 = torch.randn((self.neurons),generator=g)
    self.W2 = torch.randn((self.neurons, self.output_classes_or_neurons),generator=g)
    self.b2 = torch.randn((self.output_classes_or_neurons),generator=g)
    return [self.C, self.W1, self.b1, self.W2, self.b2]


  def final_loss(self, ):   # Heres the 2nd layer of the NN
    
    C,W1,b1,W2,b2 = self.parameters_output

    emb = C[self.X]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(emb.shape[0],self.C_iinput_size) @ W1 + b1)
    logits = h @ W2 + b2
    # indexing those 32 minibatch ground truth values with ix
    probs = F.softmax(logits,dim=1)     # Used to extract probabilities for text generation
    loss = F.cross_entropy(logits,self.Y)
    return loss.item()
    

class Make_more_v1(Make_more):
  def __init__(self, mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen=None, parameters_input=None):
    super().__init__(mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen, parameters_input)

  def initial_processor(self, ):

    # Only for initilizing and checking where the learning rate stags.
    lre = torch.linspace(-3,0,self.gen)    # the Learning rate exponent. equidistant nums. Linear
    
    self.lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
    self.lri,self.lossi = [],[]     # learning rate counts. lri is where actual lr is.
    
    self.post_lossi = []    # counts for after linear custom lr. post_lr is stationary
    self.batch_post_lossi = []


  def processor_v1(self, ):
    parameters = self.params()    # init the params(the random Cs,Ws&bs)
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))
    
    for p in parameters:
      p.requires_grad = True

    # Feels more appropriate to extract this loops outside of this class entirely
    # Reason being the learning rate decay can be tracked OR just entirely bypassed.
    for i in range(self.gen):
      # minibatch
      self.og_batch_size = self.X.shape[0]
      ix = torch.randint(0,self.og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
      emb = C[self.X[ix]]    # embed minibatch from X into C.
      
      h = torch.tanh(emb.view(emb.shape[0],self.C_iinput_size) @ W1 + b1)
      logits = h @ W2 + b2
      loss = F.cross_entropy(logits,self.Y[ix])

      # backward_pass
      for p in parameters:
        p.grad = None   # Declaring p.grad
      loss.backward()   # this derives p.grad that is used to update the params:C, W and b

      # learning rate decay:
      # Doesn't make sense to have the learning decay for the custom params(C,W&b).
      # if custom param: learning rate shoukld be previous best
      lr = self.lrs[i]
      for p in parameters:
        p.data += -lr * p.grad    # Updating the params(C,W&b) with the loss p.grad
      
      self.lri.append(lr)
      self.lossi.append(loss.item())
    
    plt.plot(self.lri,self.lossi)     # self.lri is the actual learning rate
    # plt.plot(lrei,self.lossi)
    minn = torch.tensor(self.lossi).argmin()
    print('lowest loss = ', self.lossi[minn.item()])
    self.lr = torch.tensor(self.lri)[minn].item()
    print('learning_rate = ', self.lr)

    self.parameters_output = C,W1,b1,W2,b2


  def slope_of_the_line(self,lossi,lri=None):
    
    a = torch.tensor(lossi).reshape(10,100)
    slope_of_the_line = a.diff(dim=1).float().mean(dim=1,keepdim=True)
    closest_to_zero = slope_of_the_line[slope_of_the_line < 0][-1]
    indx_by_zero = torch.where(slope_of_the_line==closest_to_zero)[0].item()
    std = a.std(dim=1).reshape(-1,1)
    print(closest_to_zero)
    plt.plot(slope_of_the_line)
    plt.plot(std)

    if lri:
      b = torch.tensor(lri).reshape(10,100)
      ideal_lr = b[indx_by_zero].mean().item()
      # take the index close sest to 0, run that index on b with mean of it. e.g b[6].mean()
        # this will give you the lr.  
      return ideal_lr
    else:
      return closest_to_zero


  def post_processor(self, custom_lr=None):
    parameters = self.parameters_output 
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))
    
    for p in parameters:
      p.requires_grad = True

    for i in range(self.gen):
      # minibatch
      self.og_batch_size = self.X.shape[0]
      ix = torch.randint(0,self.og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
      emb = C[self.X[ix]]    # embed minibatch from X into C.
      
      h = torch.tanh(emb.view(emb.shape[0],self.C_iinput_size) @ W1 + b1)
      logits = h @ W2 + b2
      loss = F.cross_entropy(logits,self.Y[ix])

      # backward_pass
      for p in parameters:
        p.grad = None   # Declaring p.grad
      loss.backward()   # this derives p.grad that is used to update the params:C, W and b

      # learning rate decay:
      # Doesn't make sense to have the learning decay for the custom params(C,W&b).
      # if custom param: learning rate shoukld be previous best
      if custom_lr:
        lr = custom_lr
      else:
        lr = self.lr
      for p in parameters:
        p.data += -lr * p.grad    # Updating the params(C,W&b) with the loss p.grad
      
      self.post_lossi.append(loss.item())
    
    plt.plot(self.post_lossi)
    lossi_tensor = torch.tensor(self.post_lossi)
    minn = lossi_tensor.argmin()
    print('lowest loss = ', self.post_lossi[minn.item()])
    print('loss mean = ', lossi_tensor.mean().item())
    print('loss std = ', lossi_tensor.std().item())
    slope_of_the_line = lossi_tensor.diff().float().mean()
    print('Slope = ', slope_of_the_line.item())
    print('c_learning_rate = ', custom_lr)
    print('lr', lr)

    self.parameters_output = C,W1,b1,W2,b2


class Make_more_v2(Make_more_v1):
  def __init__(self, mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen=None, parameters_input=None):
        super().__init__(mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen, parameters_input)


  def custom_processor(self, custom_lr=None,decay_rate=None):
    self.post_lossi = []    # Finally clearing post lossi for refills
    parameters = self.parameters_output 
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))
    
    for p in parameters:
      p.requires_grad = True

    for i in range(1,self.gen+1):
      # minibatch
      self.og_batch_size = self.X.shape[0]
      ix = torch.randint(0,self.og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
      emb = C[self.X[ix]]    # embed minibatch from X into C.
      
      h = torch.tanh(emb.view(emb.shape[0],self.C_iinput_size) @ W1 + b1)
      logits = h @ W2 + b2
      loss = F.cross_entropy(logits,self.Y[ix])

      for p in parameters:
        p.grad = None   # Declaring p.grad
      loss.backward()   # this derives p.grad that is used to update the params:C, W and b
      
      if custom_lr:
        self.lr=custom_lr
        
      if decay_rate:
        self.decay_rate = decay_rate
      else:
        self.decay_rate = 1.05
      self.batch_post_lossi.append(loss.item())
      if not i%100:
        slope_of_the_line = torch.tensor(self.batch_post_lossi).diff().float().mean()
        if slope_of_the_line > -0.0001:
          print('slope stagnating,', slope_of_the_line.item())
          print('changing lr. previous lr:', self.lr)
          self.lr /= self.decay_rate
        else:
          print('current slope =', slope_of_the_line.item())

        self.batch_post_lossi = []    # Delete the batch once slope determined
        pass

      
      for p in parameters:
        p.data += -self.lr * p.grad    # Updating the params(C,W&b) with the loss p.grad
      
      self.post_lossi.append(loss.item())
    
    plt.plot(self.post_lossi)
    lossi_tensor = torch.tensor(self.post_lossi)
    minn = lossi_tensor.argmin()
    print('lowest loss = ', self.post_lossi[minn.item()])
    print('loss mean = ', lossi_tensor.mean())
    print('loss std = ', lossi_tensor.std())
    slope_of_the_line = lossi_tensor.diff().float().mean()
    print('Slope = ', slope_of_the_line)
    print('c_learning_rate = ', custom_lr)
    print('lr', self.lr)




C = Make_more_v2(3,10,100,60,output_classes,1000)
C.initial_cats(words_processed)   # Change this to just words for old style
# C.initial_cats(words=words, og=True)    # Use this for original cats.
C.initial_processor()   # lightweight: generate pre/post and lri, lossi lists empty lists

b_path = r"C:\Users\ujasv\OneDrive\Desktop\C_pam_basic_eng.pkl"
with open(b_path, "rb") as f:
  # C = Make_more_v2(6,30,300,300,output_classes,1000)    # this was the init for params
  C.parameters_output = pickle.load(f)

C.processor_v1()        # initial lr decay

# TODO: Make it such that if last 3 mean slope is level or +ve then break the loop
for i in range(10):
  C.custom_processor(decay_rate=1.05)
C.final_loss()

# overflow temporary fix for final_loss
og_x,og_y = C.X,C.Y               
C.X,C.Y=og_x[:2000],og_y[:2000]   #temp overflow fix  
C.X,C.Y=og_x,og_y                 #back to og





# OLD CODE  V1
C = Make_more_v1(5,500,300,1000,output_classes,1000)
C.initial_cats()
C.initial_processor()   # lightweight: generate pre/post and lri, lossi lists empty lists
C.processor_v1()        # initial lr decay

# This is just to fact check that the returns match with lr in processor_v1
lr1 = C.slope_of_the_line(lossi=C.lossi, lri=C.lri)   # Check where learning rate stags

C.initial_processor()   # Clear out the lists for refills.
C.post_processor()      # custom lr 
C.slope_of_the_line(lossi=C.post_lossi)   # Check where learning rate stagnates.


# Generate from the model:
parameters = C.parameters_output      # export the params from model(C,W&b)
C_x,W1,b1,W2,b2=parameters
g = torch.Generator().manual_seed(2147483647+30)
C_iinput_size = C.mm_block_size*C.C_proj_clms
for _ in range(5):
  out = []
  context = [0] * C.mm_block_size
  while True:
    emb = C_x[torch.tensor(context)]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(1,C_iinput_size) @ W1 + b1)
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


block_size = 3# Block size around the char
X,Y = [],[]
for lines in words:
    context = [0] * block_size    # Initially 3 0's in list
    list_of_words = lines.split(' ')
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    for w in list_of_words+[itos[0]]:    # Appending . at the end of the word
        ix = stoi[w]     # Bring index of the char
        X.append(context) # Add block indexes to the inputs X
        Y.append(ix)      # Add the value to be predicted after the block, to Y
        context = context[1:] + [ix]    # Change the last index of the block to the next char
X = torch.tensor(X)
Y = torch.tensor(Y)



# Making an empty array for unique alphabets format type int32
N = torch.zeros((output_classes,output_classes), dtype=torch.int32)      

# Check if bigram is working
for lines in words:
  # alphabets of each word in a list form with leading and trailing dot `.`
  list_of_words = lines.split(' ')
  list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
  chs = ['..'] + list_of_words + ['..']     # chs is ['.', 'A', 'm', 'y', '.']
  for ch1, ch2 in zip(chs, chs[1:]):    
    bi_char = ch1 + ch2   # for 'jade' it'll probs be either 'ja' or 'ad' or 'de'
    print(bi_char)
  break


for lines in words:
  # alphabets of each word in a list form with leading and trailing dot `.`
  list_of_words = lines.split(' ')
  list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
  chs = ['..'] + list_of_words + ['..']     # chs is ['.', 'A', 'm', 'y', '.']
  for ch1, ch2 in zip(chs, chs[1:]):    
    bi_char = ch1 + ch2
    ix1 = stoi[ch1]   # getting index of the associated char-alphabet from stoi
    ix2 = stoi[ch2]
    # This will add a one to the preassigned 0 of torch.zeros in N at 
      # the row and column of the associated number of alphabet
      # Essentially collecting the occourances
    N[ix1, ix2] += 1

P = (N+1).float()     
P /= P.sum(axis=1, keepdims=True)   # P here is just the normalized N
g = torch.Generator().manual_seed(2147483647)
for i in range(5):
  out = []  
  ix = 0    
  while True:
    p = P[ix]  
    ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
    out.append(itos[ix])
    if ix == 0:   
      break
  print(''.join(out))






file = "basic_eng_min2.txt"
words_path = rf"C:\Users\ujasv\OneDrive\Desktop\{file}"    
words = open(words_path, 'r').read().splitlines()   # lines

continium=[]
for lines in words:
    list_of_words = lines.split(' ')
    # adding space at the end of each word but the last in the line
    list_of_words = [i+' ' for i in list_of_words[:-1]] + [list_of_words[-1]]
    for i in range(len(list_of_words)):
        for y in range(len(list_of_words)):
            aa = list_of_words[i:y+1]
            if aa:
                continium.append(''.join(aa))
words_processed = continium
# old code for continium

x = [i for i in words_processed if len(i.split()) >= 1]    # keep items with more than one word
set_of_chars = set(x+' '.join(x).split())     # Join and split seperates all words out by space
chars = sorted(list(set_of_chars))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['..'] = 0
itos = {i:s for s,i in stoi.items()}
output_classes = len(list(stoi.keys()))

# Attempt to parse the file in a different way
X = []
Y = []
for irr in range(len(x)):
    if irr < (len(x)-1):
        tot_previous = len(x[irr].split())
        tot_next = len(x[irr+1].split())
        if tot_previous>=tot_next:
            pass
        else:
            x_string = x[irr]
            split_xs = x_string.split() 
            X_r = [stoi[i] for i in split_xs]
            X.append(X_r)
            Y.append(stoi[(x[irr+1].split()[-1])])   # y is the next line's last word

# Check your work by collecting those itos index back to string
for a,b in zip(X,Y):
    print([itos[i] for i in a], ">>>", itos[b])

# making X dimentional, adding zeros to make it a tensor
aa = []
for irr in range(len(X)):
    if len(X[irr]) <3:
        aa.append([0]*(3-len(X[irr])) + X[irr])
    else:
        aa.append(X[irr])
X = torch.tensor(aa)