import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np


#make this example reproducible 

np.random.seed (1)

#create DataFrame

df = pd.DataFrame({'team': np.repeat(['A', 'B', 'C'], 100),
           'points': np.random.normal(loc=20, scale=2, size=300)})

#view head of DataFrame
print(df.head())

#create histogram with 20 bins
df.plot.hist(column=['points'], edgecolor='black', bins=100)
plt.show()