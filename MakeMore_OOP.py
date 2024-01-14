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
                      ):
    
    self.mm_block_size = mm_block_size
    self.C_proj_clms = C_proj_clms
    self.neurons = neurons
    self.mini_batch_size = mini_batch_size
    self.output_classes_or_neurons = output_classes_or_neurons
    self.gen = gen
    self.total_params = None
    self.X = None
    self.C, self.W1, self.b1, self.W2, self.b2 = [None]*5

  def blocker(self, mm_block_size):
      block_size = mm_block_size    # Block size around the char
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
      self.X = X
      self.Y = Y
      return X,Y

  def juice(self):
    mm_block_size = int(self.mm_block_size)
    C_proj_clms = int(self.C_proj_clms)
    C_iinput_size =  C_proj_clms * mm_block_size
    neurons = int(self.neurons)
    output_classes_or_neurons = int(self.output_classes_or_neurons)
    mini_batch_size = int(self.mini_batch_size)
    unique_items = len(list(stoi.keys()))

    X_Y = self.blocker(mm_block_size)
    X = X_Y[0]      # column is mm_block_size
    Y = X_Y[1]
    batch_size = X.shape[0]
    
    def params(C_proj_clms,neurons,output_classes_or_neurons):
        g = torch.Generator().manual_seed(2147483647)
        
        C = torch.randn((unique_items,C_proj_clms),generator=g)
        iinput_size = mm_block_size
        C_iinput_size =  C_proj_clms*iinput_size
        
        W1 = torch.randn((C_iinput_size,neurons),generator=g)
        b1 = torch.randn((neurons),generator=g)
        W2 = torch.randn((neurons, output_classes_or_neurons),generator=g)
        b2 = torch.randn((output_classes_or_neurons),generator=g)
        return [C, W1, b1, W2, b2]

    parameters = params(C_proj_clms,neurons,output_classes_or_neurons)
    self.C, self.W1, self.b1, self.W2, self.b2 = (parameters[i] for i in range(len(parameters)))
    C,W1,b1,W2,b2 = self.C, self.W1, self.b1, self.W2, self.b2    # this is a work around to access these yet not break the code

    self.total_params = sum(p.nelement() for p in parameters)
    print('total params size:', self.total_params)

    for p in parameters:
      p.requires_grad = True

    if not self.gen:      # generations are declared.
       self.gen = 1000
    
    lre = torch.linspace(-3,0,self.gen)    # the Learning rate exponent. equidistant nums
    lrs = 10**lre     # neatly makes lre non-linear between 0.001 and 1
    lrei,lri,lossi = [],[],[]     # learning rate counters

    for i in range(self.gen):
      # minibatch
      ix = torch.randint(0,batch_size,(mini_batch_size,))    #  generate indexes
      
      emb = C[X[ix]]    # embeddings are essentially inputs
      
      h = torch.tanh(emb.view(emb.shape[0],C_iinput_size) @ W1 + b1)
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

    emb = C[X]    # embeddings are essentially inputs
    h = torch.tanh(emb.view(emb.shape[0],C_iinput_size) @ W1 + b1)
    logits = h @ W2 + b2
    # indexing those 32 minibatch ground truth values with ix
    probs = F.softmax(logits,dim=1)
    loss = F.cross_entropy(logits,Y)
    return loss.item()
    
    # plt.plot(lri,lossi)
    # plt.plot(lrei,lossi)


def mm_loss(block_size,i_C_prjs,i_neurons_val,i_batch_size_val, i_output_classes,
            gen=None):
  C = Make_more(
                mm_block_size=block_size,
                C_proj_clms=i_C_prjs,
                neurons=i_neurons_val,
                mini_batch_size=i_batch_size_val,
                output_classes_or_neurons=i_output_classes,
                gen=gen
                      )
  final_loss = C.juice()
  parms = C.total_params
  return final_loss, parms

# Use this for testing return a base loss
l,p = mm_loss(2,2,30,30,27)

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
                            )
  losses_and_their_associatetd_params.append([final_loss,i_size,i_C_prjs,i_neurons_val.item(),i_batch_size_val.item()])
  print(parms,losses_and_their_associatetd_params)

# Dump the losses
b_path = r"C:\Users\ujasv\OneDrive\Desktop\losses.pkl"
with open(b_path, "wb") as f:
  pickle.dump(losses_and_their_associatetd_params, f)
  print('just dumped')

# Call the losses
b_path = r"C:\Users\ujasv\OneDrive\Desktop\losses.pkl"
with open(b_path, "rb") as f:
  losses_and_their_associatetd_params = pickle.load(f)  
aa = torch.tensor(losses_and_their_associatetd_params)

print('Lowest loss params', aa[aa[:,0].argmin().item()])


# more the neurons higher the loss, 
# bigger the minibatch size lower the loss
fig,ax = plt.subplots()
aax = aa/torch.sum(torch.tensor(aa),dim=0,keepdim=True)

ax.set_xlim(500,600)
ax.set_ylim(0.00005,0.0009)
ax.plot(aax,label=range(aax.shape[1]))
ax.legend()