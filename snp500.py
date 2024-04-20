import torch
import matplotlib.pyplot as plt
import numpy as np

# Working animation model with matplotloib that plots and clears in a loop

import pandas as pd
snp = r"C:\Users\ujasv\Downloads\snp500 - Sheet3.csv"

df = pd.read_csv(snp)
tik = list(df.columns) # All ticker symbols

# Dropping I suppose the first column or row
df = df.drop(index=0)
# aaa = pd.to_numeric(df,errors='coerce')
arr = np.array(df)
arr = arr.astype(float)
t_arr = torch.tensor(arr)
t_arr = t_arr[:,1:]       # Remove first nan column

# the tuple returned here is weird.. tuple[1][0] is 0 that seems to signify column
# TODO: investigate this. Its an important concept
nan_indexes = torch.where(t_arr.isnan())

# Check values that are nan's
# t_arr.isnan()

# Making nan's 0's
t_arr[nan_indexes]=0

# finding the mean per column of all columns to find middle number per stock.
tx_mean = t_arr.mean(axis=0)

# returning indexes of lowest price stock to highest price stock(mean val) left to right.
t_arr = t_arr[:,tx_mean.argsort()]
# Not sure what this is doing.. maybe trimming off null or zero vals.
t_arr= t_arr[:,50:]


# Guess you can do either mean sorting or this std sorting
# finding the std per column- per stock
t_arr_std = t_arr.std(axis=0)

# returning indexes of lowest growth(std) to highest left to right.
t_arr = t_arr[:,t_arr_std.argsort()]



tx_log = torch.log(t_arr) 
# log of 0 is -inf and hence zeros will show up as -inf.
# TODO: now find std and overall slope of the data. sort the steeper slope to rightmost and shallowe to leftmost
# t_arr/=t_arr.sum(axis=0,keepdim=True)

# std for all columns:
# aa = torch.std(t_arr,dim=0)


# returs all column index that has nan type.
# torch.where(torch.isnan(aa))


# use either t_arr or tx_log and make sure to set y limit appropriately.
array_to_be_used = t_arr

plt.ion() # enable interactive mode
fig, ax = plt.subplots()

# print(t_arr.shape)
mods = [i for i in range(2500) if not i % 2]
print("total mods:",len(mods),"sample:",mods[:10] )
for i in mods: 
    # print('here', i)

    # [:,0] - the rows are the data points and [0,:] columns are the datasets of different colors
    # x axis extends to the len(data[:,0]), y axis is the plot of those values 
    # dumm = t_arr[i,:]
    
    # Clear old plot
    ax.clear()  
    
    # sets y axis limits. Otherwise the limits are the top datapoint.
    ax.set_ylim(0,10)       # Suitable for natural log array
    # ax.set_ylim(0,400)       # Suitable for raw array t_arr

    # ax.set_xlim(0,5)      # xlim  isnt useful because thats the quantity of data and thats mostly constant.
    
    to_display = array_to_be_used[i]

    point_size = [6]*to_display.shape[0]
    ax.scatter(torch.arange(to_display.shape[0]),to_display,s=point_size,)

    # ax.hist(dumm)
    
    # Refresh plot 
    fig.canvas.draw()   
    plt.pause(0.01)

plt.show()
