
import numpy as np
import matplotlib.pyplot as plt
import torch


# Finding slope(trending up or down) of a tensor array
b = torch.randn((20))
torch.diff(b).float().mean()
# Checkout Makemore.py for use case: finding where learning rate
    # stagnates the losses/starts flattening after diminishing losses.

# Get the column that needs to be sorted, return indexes from lowest to highest
# then call that on the original array to sort it by that particular column
arg_sort_indexes = torch.argsort(arr[:,0])
sorted_by_column = arr[arg_sort_indexes]

# ___________________________________________________________

# 3 lists containing 4 items. Shape of this is (2,4). matrix of 3 rows and 4 columns
# rand() produces normal randown as compared to randn() which produces  gaussian distribution
arr = np.random.rand(3,4) 

# empty array, same as  zeros/fake_vals. First argument is the shape hence double bracs. second is dtype
np.empty((2,3,4),dtype=float)

# 3D Array of 2 lists each of those lists contains 3 items,
    # and each of those items contains 4 items.
    # Dimension 0 has 2 items,  dim1 has 3 items and dimension 2 has 4.
arr = np.random.rand(2,3,4)     # numbers between 0 and 1.
arr.size                        # total items in the array. 2 * 3 * 4. In this case 24.
arr.ndim                        # total dimensions in the array. len(arr.shape). In this case 3 since its a 3D array.
arr.shape                       # the shape(2,3,4).
np.random.randint(1,5,size=(2,4))      # This one returns rand ints between(inclusive) 1 and 4 with shape 2,4

arr.reshape(12,2)      # 12 rows 2 columns. arr.size should be rows*column. in this case 24

# this makes a 
    # reshape it to 2 rows and 3 columns. simply 2 times 3
rows = 2
columns = 3
arr = np.arange(-1,3,0.25)      # Vector of numbers between -1 to 3 stepping by 0.25,
# calling an index is calling the dimension of the array
arr = arr[:(rows*columns)]      # trim the vector upto a [:i]
arr.reshape(rows,columns)       # since there are 6 items in the vector, row*column should be 6

# 30 equidistant numbers between -5 and 5. reshape it such that axis
arr = np.linspace(-5,5,30).reshape((5,2,3))    
# move columns to rows.
# args esentially act as index/dimension of the array. (0,row,column) becomes (0,column,row).
arr.transpose(0,2,1)    

# find a value within an array: in this case find index of the max value
arr = np.array([3,2,1,3,2])
np.where(arr == np.max(arr))

# return index of the maximum value.
np.argmax(arr)
# keeps dimensions intact:
np.argmax(arr,axis=1,keepdims=True)

x_vals = arr[:,0]         # says all rows and 0th column
y_vals = arr[:,1]         # all rows, 1st column
x_values, y_values = arr[:,0], arr[:,1]     # : is all rows and 0th column; 1st column

# switching columns around
arr = np.array([[1, 2], [3, 4], [5, 6]])
arr[:, [0, 1]] = arr[:, [1, 0]]

# flip array left to right:
arr[::-1]
np.flip(arr)

# flip array upside down.
    # take mean, subtract mean from the data
    # then do -1 times that array that makes positive numbers -ve and -ve nums positive
-1 * (arr - np.mean(arr))

# normalize
aa=arr
aax = aa/np.sum(aa,axis=0,keepdims=True)      # Normalize using numpy np
aax = aa/torch.sum(torch.tensor(aa),dim=1,keepdim=True) # Normalize using torch


# ______________________________________________________________________
# ______________________________________________________________________

# Matplotlib
import matplotlib.pyplot as plt

arr = np.random.randint(-1,1,99)
x_vals = range(len(arr))
y_vals = arr

size_of_points = [1]*len(arr)       # If there are too much data you can squeeze points to make it visible
plt.scatter(x_vals,y_vals,size_of_points)

plt.plot(y_vals)

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
aa = np.random.randn(111,4)

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
        plt.text(j, i, N[i, j].item(), ha="center", va="top", color='gray')
plt.axis('off')
plt.show()




