
import matplotlib.pyplot as plt
import numpy as np

# Working animation model with matplotloib that plots and clears in a loop

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
