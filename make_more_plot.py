
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import torch

words_path = r"C:\Users\ujasv\Downloads\names.txt"
words = open(words_path, 'r').read().splitlines()   # list of words
set_of_chars = set(''.join(words)) 
chars = sorted(list(set_of_chars))

stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}

total_uniques = len(list(stoi.keys()))
N = torch.zeros((total_uniques,total_uniques), dtype=torch.int32)      

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
inputs,neurons = 6,100    # neurons are arbitrary
W1 = torch.randn((inputs,neurons),generator=g)
b1 = torch.randn((neurons),generator=g)
emb = C[X]    # embeddings are essentially inputs
h = torch.tanh(emb.view(emb.shape[0],6) @ W1 + b1)

W2 = torch.randn((100, 27),generator=g)
b2 = torch.randn((27),generator=g)
logits = h @ W2 + b2
loss = F.cross_entropy(logits,Y)



plt.ion() # enable interactive mode
fig, ax = plt.subplots()

for i in range(20):
    print('here', i)

    # [:,0] - the rows are the data points and [0,:] columns are the datasets of different colors
    # x axis extends to the len(data[:,0]), y axis is the plot of those values 
    dumm = np.random.randn(250,2)        # 0 dim is the row 
    
    # Clear old plot
    ax.clear()  
    
    # sets y axis limits. Otherwise the limits are the top datapoint.
    ax.set_ylim(-5,5)
    # ax.set_xlim(0,5)      # xlim  isnt useful because thats the quantity of data and thats mostly constant.
    
    ax.plot(dumm)
    
    # Refresh plot 
    fig.canvas.draw()   
    plt.pause(0.1)

plt.show()

