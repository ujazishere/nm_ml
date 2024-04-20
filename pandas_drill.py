import pandas as pd
import numpy as np

appl_data = r"C:\Users\ujasv\OneDrive\Desktop\codes\HOOD_Dataset.csv"

df = pd.read_csv(appl_data)



tik = list(df.columns) # All column heads(ticker symbols) in list form

df.describe()       # gives typical statistics of the data

# excludes 1st column from dataframe, replaces all dollar amounts to float
X = df[df.columns[3:]].replace({'\$':""}, regex=True).astype(float)
y = df['Close/Last'].replace({'\$': ''}, regex=True).astype(float)
y = y.to_frame()        # Converting series df to frame for columns




# Getting unique nationalities and associating numbers to each items.
nat_uniq = pd.unique(df['Nationality'])
nat_nums_dict = dict(zip(nat_uniq, range(len(nat_uniq))))

# Getting unique of 'Go' items then making dict of items and associating number
Go_uniq = pd.unique(df['Go'])
Go_uniq_dict = dict(zip(Go_uniq, range(len(Go_uniq))))

# Replacing those 'Go' items with associated numbers
df['Go'] = df['Go'].map(Go_uniq_dict)




snp = r"C:\Users\ujasv\Downloads\snp500 - Sheet3.csv"

df = pd.read_csv(snp)
df = df[1:]         # removing first row since it contains "Close"
df = df.astype(float)        # Convert df data to float
df.describe()

