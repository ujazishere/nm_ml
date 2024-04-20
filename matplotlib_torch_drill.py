import numpy as np
import matplotlib.pyplot as plt
import torch

# Finding slope(trending up or down) of a tensor array
arr = torch.rand((2,3,4))
b = torch.randn((20))
torch.diff(b).float().mean()
# Checkout Makemore.py for use case: finding where learning rate
    # stagnates the losses/starts flattening after diminishing losses.

# flip array left to right or top to bottom
# try this for 2D array the columns gets flipped upside down.
# arr = torch.arange(1,13,1).reshape(6,2)
arr = torch.arange(1,13,1)
torch.flip(arr,dims=[0])

# the column that needs to be sorted passed as argument, argsort returns just sorted indexes of that particular column
arg_sort_indexes = torch.argsort(arr[:,0])
# that column of sorted indexes are passed in as index returns, by rows, and hence returns the whole row when passed a particular index..
# sorting the whole array by a particular column.
sorted_by_column = arr[arg_sort_indexes]

# normalize
aa = arr
aax = aa/torch.sum(torch.tensor(aa),dim=1,keepdim=True) # Normalize using torch
# Normalize using Andrej Karparthy method
xa = (aa - aa.mean())/torch.sqrt(aa.var()+1e-5)

# Non-linear downward slope 
# When x/= number is closer to 1 the slope is shallower.
y = []
x = 2
for i in range(50):
    x/=1.3
    y.append(x)
aa = [val for val in y if val<0.1 and val>0.01]
plt.plot(aa)

# ______________________________________________________________________
# ______________________________________________________________________

# Matplotlib
import matplotlib.pyplot as plt

# squeeze poins to show all data 
arr = np.random.randint(-1,1,99)
x_vals = range(len(arr))
y_vals = arr

size_of_points = [1]*len(arr)       # If there are too much data you can squeeze points to make it visible
plt.scatter(x_vals,y_vals,size_of_points)

plt.plot(y_vals)

tx = torch.rand((2500,500))     # these can be snp500 stock prices for 10 years.
plt.plot(tx[:,3][:2000])        # data sample of a particular ticker. 3rd column in this case and upto 2000 data points

# The higher the mean the the closer this return is to zero and so for mean 1 and above the return is between 0 and 1. as the mean gets below 1..
# the return goes above 1. The f(x) is inversly proportional to x of some sort here.
# Use case: for varied data for too high or low scales it helps bring data close together for visualization. 
# In the case of snp500 some data points range between 3000 and 4000 and others between 0 and 5.
# Better use log in that case but this is also one way to do it.
one_over_mean= 1/tx[:,3].mean().item()     
# multiplied with the function stretches the data top to bottom.
plt.plot(tx[:,3][:2000]**one_over_mean)       
# Compare above with this log of data to squish it down for visibility. Do to all data and compare result.
plt.plot(torch.log(tx[:,3][:2000]))
# ______________________________________________________________________
# Working animation model with matplotloib that plots 2D arrays
# (columns are daatasets) check best_model_appl.py for use case.

plt.ion() # enable interactive mode
fig, ax = plt.subplots()
pause = 0.1

for i in range(20):
    print('here', i)

    dumm = np.random.randn(250,2)        # 0 dim is the row 
    
    # Clear old plot
    ax.clear()  
    
    # x limit is stable so no need to set x limit since len of all rows and nth column is always same
    # unless you want to zoom to specific area on x axis
    ax.set_xlim(159,170)      
    # sets y axis limits. Otherwise the limits are the top datapoint.
    ax.set_ylim(-5,5)
    # ax.set_xlim(0,5)      # xlim  isnt useful because thats the quantity of data and thats mostly constant.

    # columns are the differnt sets of data and rows are the datapoints on y axis themselves
    # [:,0] - the rows are the data points and [0,:] columns are the datasets of different colors
    # x axis extends to the len(data[:,0]), y axis is the plot of those values 
    # np.hstack stacks columns next to each other
    # transposed_d = softmax_outputs +
    y = np.random.randn(111,3)
    softmax_outputs = np.random.randn(111,2)*1.2
    data_to_plot = np.hstack((y,softmax_outputs))       # First arg itself is the column bunch hence double bracs
    ax.plot(data_to_plot)   # plotting the data

    # Refresh plot 
    fig.canvas.draw()   
    plt.pause(0.1)

plt.show()

# ____________________________________________________

def pst(y_vals):
    if y_vals.ndim == 2:            # If just rows and columns then
        x = range(y_vals.shape[0])
        for i in range(y_vals.shape[1]):    # iterate over all columns.
            plt.scatter(x, y_vals[:,i])
    if y_vals.ndim == 3:    # could do <= as well
        x = range(y_vals.shape[0])
        for i in range(y_vals.shape[0]):
            for a in range(y_vals.shape[2]):
                plt.scatter(x, y_vals[i,:,a])
# ______________________________________________________

# line plotting 4 columns as datasets.
fig,ax = plt.subplots()
aa = torch.randn((111,4))

# aax = aa/np.sum(aa,axis=0,keepdims=True)      # Normalize using numpy np
# aax = aa/torch.sum(torch.tensor(aa),dim=1,keepdim=True) # Normalize using torch

ax.plot(aa, label=range(aa.shape[1]))   # aa columns are datasets(y vals). label shows columns
ax.legend()
plt.show()

# _____________________________________________________

words_path = r"C:\Users\ujasv\Downloads\names.txt"
words = open(words_path, 'r').read().splitlines()   # list of words

# collapses all words to a consistent string, then making a set to get uniques alphabets only
set_of_chars = set(''.join(words)) 
# sorted alphabets in a list form
chars = sorted(list(set_of_chars))

# Assigning numbers to each alphabet starting from 1, hencee i+1. Alphabets as keys and numbers as vals
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
# Flipping keys to values and values to keys.
itos = {i:s for s,i in stoi.items()}

total_uniques = len(list(stoi.keys()))
# Making an empty array for unique alphabets format type int32
N = torch.zeros((total_uniques,total_uniques), dtype=torch.int32)      
N = torch.zeros((total_uniques,total_uniques), dtype=torch.int32)      
# show fig
for w in words:
  # alphabets of each word in a list form with leading and trailing dot `.`
  chs = ['.'] + list(w) + ['.']     # chs is ['.', 'A', 'm', 'y', '.']
  for ch1, ch2 in zip(chs, chs[1:]):    # first iter will be: ch1 as ['.'] and ch2 ['A']
    bi_char = ch1 + ch2   # for 'jade' it'll probs be either 'ja' or 'ad' or 'de'
    ix1 = stoi[ch1]   # This is the index of the char-alphabet from stoi
    ix2 = stoi[ch2]
    # This will add a one to the preassigned 0 of torch.zeros in N at 
      # the row and column of the associated number of alphabet
      # Essentially collecting the occourances
    N[ix1, ix2] += 1

# Show the uniiques and their occourances in a presentable manner(optional)
total_uniques = len(list(stoi.keys()))
plt.figure(figsize=(total_uniques,total_uniques))
plt.imshow(N, cmap='Blues')
for i in range(total_uniques):
    for j in range(total_uniques):
        chstr = itos[i] + itos[j]   # gets first and seccond char alphabets and stores it here
        
        # initial j,i args are the x and y axis assignments. y axis goes from top to bottom unlike bottom to top
          # increments of it are like (0,0), (1,0), (2,0) ...
        plt.text(j, i, chstr, ha="center", va="bottom", color='gray')
        plt.text(j, i, N[i, j].item(), ha="center", va="top", Color='gray')
plt.axis('off')
plt.show()




