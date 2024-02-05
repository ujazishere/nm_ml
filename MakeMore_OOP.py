import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
from itertools import product
import pickle

words_path = r"C:\Users\ujasv\Downloads\names.txt"
words = open(words_path, 'r').read().splitlines()   # list of words
set_of_chars = set(''.join(words)) 
chars = sorted(list(set_of_chars))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

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
    
    if not self.gen:      # generations here since its clogging processor
       self.gen = 1000
       
    # Not necessary to declare outputs to be used outside of class.
      # Unless using to create bool for bypassing entensive processing
    self.parameters_output = None

  def initial_cats(self, ):      # call this to declare X and Y
      block_size = self.mm_block_size    # Block size around the char
      X,Y = [],[]
      for w in words:
          context = [0] * block_size    # Initially 3 0's in list
          for ch in w + '.':    # Appending . at the end of the word. This is the index 1 seperator in stoi
              ix = stoi[ch]     # Bring index of the char
              X.append(context) # Add block indexes to the inputs X
              Y.append(ix)      # Add the value to be predicted after the block, to Y
              context = context[1:] + [ix]    # Change the last index of the block to the next char
      X = torch.tensor(X)
      Y = torch.tensor(Y)
      self.X = X
      self.Y = Y


  def params(self, ):
    self.C_iinput_size =  self.C_proj_clms * self.mm_block_size
    self.unique_items = len(list(stoi.keys()))

    g = torch.Generator().manual_seed(2147483647)
    
    self.C = torch.randn((self.unique_items,self.C_proj_clms),generator=g)
    
    self.W1 = torch.randn((self.C_iinput_size,self.neurons),generator=g)
    self.b1 = torch.randn((self.neurons),generator=g)
    self.W2 = torch.randn((self.neurons, self.output_classes_or_neurons),generator=g)
    self.b2 = torch.randn((self.output_classes_or_neurons),generator=g)
    return [self.C, self.W1, self.b1, self.W2, self.b2]


  def processor(self, ):
    if self.parameters_input:
      parameters = self.parameters_input
    else:   # adding learning rate
      parameters = self.params()
      # Only for initilizing and checking where the learning rate stags.
      lre = torch.linspace(-3,0,self.gen)    # the Learning rate exponent. equidistant nums. Linear
      lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
      lrei,lri,lossi = [],[],[]     # learning rate counts. lri is where actual lr is.

    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))

    self.total_params = sum(p.nelement() for p in parameters)

    for p in parameters:
      p.requires_grad = True

    # Feels more appropriate to extract this loops outside of this class entirely
    # Reason being the learning rate decay can be tracked OR just entirely bypassed.
    for i in range(self.gen):
      # minibatch
      og_batch_size = self.X.shape[0]
      ix = torch.randint(0,og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
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
      if not self.parameters_output:
        lr = lrs[i]
      else:
        lr = self.lr
      for p in parameters:
        p.data += -lr * p.grad    # Updating the params(C,W&b) with the loss p.grad
      
      if not self.parameters_output:
        lrei.append(lre[i])
        lri.append(lr)
        lossi.append(loss.item())
    
    if not self.parameters_output:
      plt.plot(lri,lossi)     # lri is the actual learning rate
      # plt.plot(lrei,lossi)
      minn = torch.tensor(lossi).argmin()
      print('lowest loss = ', lossi[minn.item()])
      self.lr = torch.tensor(lri)[minn].item()
      print('learning_rate = ', self.lr)

    self.parameters_output = C,W1,b1,W2,b2

  def final_loss(self, ):   # Heres the 2nd layer of the NN
    
    C,W1,b1,W2,b2 = self.parameters_output

    emb = C[self.X]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(emb.shape[0],self.C_iinput_size) @ W1 + b1)
    logits = h @ W2 + b2
    # indexing those 32 minibatch ground truth values with ix
    probs = F.softmax(logits,dim=1)     # Used to extract probabilities for text generation
    loss = F.cross_entropy(logits,self.Y)
    return loss.item()
    

  def juice(self):
    # These are all ints because makeing it self. makes them other type for some reason.
    mm_block_size = int(self.mm_block_size)
    C_proj_clms = int(self.C_proj_clms)
    C_iinput_size =  C_proj_clms * mm_block_size
    neurons = int(self.neurons)
    output_classes_or_neurons = int(self.output_classes_or_neurons)
    mini_batch_size = int(self.mini_batch_size)
    unique_items = len(list(stoi.keys()))

    self.initial_cats()
    X = self.X
    Y = self.Y
    og_batch_size = X.shape[0]
    
    def params(C_proj_clms,neurons,output_classes_or_neurons):    # inner function within juice
        g = torch.Generator().manual_seed(2147483647)
        
        C = torch.randn((unique_items,C_proj_clms),generator=g)
        iinput_size = mm_block_size
        C_iinput_size =  C_proj_clms*iinput_size
        
        W1 = torch.randn((C_iinput_size,neurons),generator=g)
        b1 = torch.randn((neurons),generator=g)
        W2 = torch.randn((neurons, output_classes_or_neurons),generator=g)
        b2 = torch.randn((output_classes_or_neurons),generator=g)
        return [C, W1, b1, W2, b2]
    
    # To take in preassigned C,W&b vals to build up on the previous loss
    if self.parameters_input:
      parameters = self.parameters_input
    else:
      parameters = params(C_proj_clms,neurons,output_classes_or_neurons)
    # self.C, self.W1, self.b1, self.W2, self.b2 = (parameters[i] for i in range(len(parameters)))
    # C,W1,b1,W2,b2 = self.C, self.W1, self.b1, self.W2, self.b2    # this is a work around to access these yet not break the code
    
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))

    self.total_params = sum(p.nelement() for p in parameters)
    print('total params size:', self.total_params)

    for p in parameters:
      p.requires_grad = True

    if not self.gen:      # generations are declared.
       self.gen = 1000
    
    lre = torch.linspace(-3,0,self.gen)    # the Learning rate exponent. equidistant nums
    lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
    # lrei,lri,lossi = [],[],[]     # learning rate counters

    # Feels more appropriate to extract this loops outside of this class entirely
    # Reason being the learning rate decay can be tracked OR just entirely bypassed.
    for i in range(self.gen):
      # minibatch
      ix = torch.randint(0,og_batch_size,(mini_batch_size,))    #  generate indexes
      
      emb = C[X[ix]]    # embeddings are essentially inputs
      
      h = torch.tanh(emb.view(emb.shape[0],C_iinput_size) @ W1 + b1)
      logits = h @ W2 + b2
      loss = F.cross_entropy(logits,Y[ix])

      # backward_pass
      for p in parameters:
        p.grad = None   # Declaring p.grad
      loss.backward()   # this derives p.grad that is used to update the params:C, W and b

      # learning rate decay:
      # Doesn't make sense to have the learning decay for the custom params(C,W&b).
      # if custom param: learning rate shoukld be 
      lr = lrs[i]
      for p in parameters:
        p.data += -lr * p.grad    # Updaating the params(C,W&b) with the loss p.grad
      
      # lrei.append(lre[i])
      # lri.append(lr)
      # lossi.append(loss.item())

    emb = C[X]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(emb.shape[0],C_iinput_size) @ W1 + b1)
    logits = h @ W2 + b2
    # indexing those 32 minibatch ground truth values with ix
    probs = F.softmax(logits,dim=1)     # Used to extract probabilities for text generation
    loss = F.cross_entropy(logits,Y)
    return loss.item()
    
    # plt.plot(lri,lossi)
    # plt.plot(lrei,lossi)

class Make_more_v1(Make_more):
  def __init__(self, mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen=None, parameters_input=None):
    super().__init__(mm_block_size, C_proj_clms, neurons, mini_batch_size, output_classes_or_neurons, gen, parameters_input)

  def initial_processor(self, ):

    # Only for initilizing and checking where the learning rate stags.
    lre = torch.linspace(-3,0,self.gen)    # the Learning rate exponent. equidistant nums. Linear
    
    self.lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
    self.lri,self.lossi = [],[]     # learning rate counts. lri is where actual lr is.
    
    self.post_lossi = []    # counts for after linear custom lr. post_lr is stationary


  def processor_v1(self, ):
    parameters = self.params()    # init the params(the random Cs,Ws&bs)
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))
    
    for p in parameters:
      p.requires_grad = True

    # Feels more appropriate to extract this loops outside of this class entirely
    # Reason being the learning rate decay can be tracked OR just entirely bypassed.
    for i in range(self.gen):
      # minibatch
      og_batch_size = self.X.shape[0]
      ix = torch.randint(0,og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
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


  def post_processor(self, custom_lr=None):
    parameters = self.parameters_output 
    C,W1,b1,W2,b2 = (parameters[i] for i in range(len(parameters)))
    
    for p in parameters:
      p.requires_grad = True

    for i in range(self.gen):
      # minibatch
      og_batch_size = self.X.shape[0]
      ix = torch.randint(0,og_batch_size,(self.mini_batch_size,))    #  generate indexes for minibatch
      
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
    print('loss mean = ', lossi_tensor.mean())
    slope_of_the_line = lossi_tensor.diff(dim=1).float().mean(dim=1,keepdim=True)
    print('Slope = ', slope_of_the_line)
    print('loss std = ', lossi_tensor.std())
    print('c_learning_rate = ', custom_lr)
    print('lr', lr)

    self.parameters_output = C,W1,b1,W2,b2




C = Make_more_v1(2,4,300,300,27,1000)
C.initial_cats()
C.initial_processor()   # lightweight: generate pre/post and lri, lossi lists empty lists
C.processor_v1()        # initial lr decay

# This is just to fact check that the returns match with lr in processor_v1
lr1 = C.slope_of_the_line(lossi=C.lossi, lri=C.lri)   # Check where learning rate stags

C.initial_processor()   # Clear out the lists for refills.
C.post_processor()      # custom lr 
C.slope_of_the_line(lossi=C.post_lossi)   # Check where learning rate stags


# Generate from the model:
parameters = C.parameters_output      # export the params from model(C,W&b)
C_x,W1,b1,W2,b2=parameters
g = torch.Generator().manual_seed(2147483647+30)
C_iinput_size = C.mm_block_size*C.C_proj_clms
for _ in range(20):
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









    
# WIP This is a productive trial version that discards use of the juice,
  # permutations and combinations hasnt been accounted for.
def mm_loss_trial(block_size,i_C_prjs,i_neurons_val,i_batch_size_val, i_output_classes,
            gen=None,parameters_input=None):
  C = Make_more(
                mm_block_size=block_size,
                C_proj_clms=i_C_prjs,
                neurons=i_neurons_val,
                mini_batch_size=i_batch_size_val,
                output_classes_or_neurons=i_output_classes,
                gen=gen,
                parameters_input=parameters_input
                      )
  return C
C = mm_loss_trial(2,4,300,300,27,1000)

C.initial_cats()      # derive X and Y

# Layer 1 and learning/rate and decay. if initialized it derives params and plot.
# and determines the learning rate based on that.
C.processor()         
C.final_loss()    # Second layer of the NN fairly unprocessed with no backprop.

C.parameters_input =  C.parameters_output     # feeding in the C,W&b back   
C.processor()
C.final_loss()

C.lr = C.lr/1.2         # Decaying learning rate ever so slightly
C.parameters_input =  C.parameters_output
C.processor()
C.final_loss()



def mm_loss(block_size,i_C_prjs,i_neurons_val,i_batch_size_val, i_output_classes,
            gen=None,parameters_input=None):
  C = Make_more(
                mm_block_size=block_size,
                C_proj_clms=i_C_prjs,
                neurons=i_neurons_val,
                mini_batch_size=i_batch_size_val,
                output_classes_or_neurons=i_output_classes,
                gen=gen,
                parameters_input=parameters_input
                      )
  final_loss = C.juice()
  parms = C.total_params
  return final_loss, parms

# Use this for testing return a base loss
C = mm_loss(2,2,30,30,27)

i_size_of_input_block =i_blks = range(2,5)
i_prj_clms = range(2,10)
i_neurons = 1.2**torch.arange(25,35)
# i_neurons = 1.08**torch.arange(60,90)
i_neurons = i_neurons.type(torch.int)
i_batch_size = i_neurons/3
i_batch_size = i_batch_size.type(torch.int) 

# Permutations and Combinations. Just using upto 4 items othersie it blows up
# len(list(product(i_size_of_input_block,i_prj_clms[:4],i_neurons[:4],i_batch_size[:4])))
len(list(product(i_size_of_input_block,i_prj_clms,i_neurons,i_batch_size)))

losses_and_their_associatetd_params = []
# cannot access the variable for doing the for loop since its a tuple.
# Hence have to use the whole sentence of `product()`
# mind the neuron_vals and mini_batch_vals these are interior variables, hence vals.
# mind the .item() to get get only the ints rather than tensor type.
for i_size, i_C_prjs,i_neurons_val,i_batch_size_val in product(i_size_of_input_block,i_prj_clms,i_neurons,i_batch_size):
  final_loss,parms = mm_loss(block_size=i_size,
                        i_C_prjs=i_C_prjs,
                        i_neurons_val=i_neurons_val.item(),
                        i_batch_size_val=i_batch_size_val.item(),
                        i_output_classes=27,
                        gen=1000
                            )
  losses_and_their_associatetd_params.append([final_loss,i_size,i_C_prjs,i_neurons_val.item(),i_batch_size_val.item()])
  print(final_loss)





# *****CAUTION*****
# Dump the losses(HARD WRITE)
b_path = r"C:\Users\ujasv\OneDrive\Desktop\losses.pkl"
with open(b_path, "wb") as f:
  pickle.dump(losses_and_their_associatetd_params, f)
  print('just dumped')

# Call the losses(READ)
b_path = r"C:\Users\ujasv\OneDrive\Desktop\losses.pkl"
with open(b_path, "rb") as f:
  losses_and_their_associatetd_params = pickle.load(f)  
aa = torch.tensor(losses_and_their_associatetd_params)

print('Lowest loss params', aa[aa[:,0].argmin().item()])


# more the neurons higher the loss, 
# bigger the minibatch size lower the loss, more the processing time.
fig,ax = plt.subplots()
aax = aa/torch.sum(torch.tensor(aa),dim=0,keepdim=True)

ax.set_xlim(500,600)
ax.set_ylim(0.00005,0.0009)
ax.plot(aax,label=range(aax.shape[1]))
ax.legend()