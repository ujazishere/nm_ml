import torch
import matplotlib.pyplot as plt
import numpy as np

# Working animation model with matplotloib that plots and clears in a loop

import pandas as pd
snp = r"C:\Users\ujasv\Downloads\snp500 - Sheet3.csv"

df = pd.read_csv(snp)
aaa = pd.to_numeric(df,errors='coerce')
arr = np.array(df)
xx = arr[1:2700,1:500]
xxx = xx.astype(float)
tx = torch.tensor(xxx)

# std for all columns:
aa = torch.std(tx,dim=0)

# returs all column index that has nan type.
torch.where(torch.isnan(aa))





async def get_tasks(session):
    tasks = []
    for airport_id in all_datis_airports:
        url = f"https://datis.clowd.io/api/{airport_id}"
        tasks.append(asyncio.create_task(session.get(url)))
    return tasks

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = await get_tasks(session)
        # Upto here the tasks are created which is very light.

        # Actual pull work is done using as_completed 
        datis_resp = []
        for task in asyncio.as_completed(tasks):        # use .gather() instead of .as_completed for background completion
            resp = await task 
            jj = await resp.json()
            datis_raw = 'n/a'
            if type(jj) == list and 'datis' in jj[0].keys():
                datis_raw = jj[0]['datis']
            datis_resp.append(datis_raw)
        return datis_resp

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
