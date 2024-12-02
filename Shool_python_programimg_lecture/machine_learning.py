import numpy as np
import matplotlib.pyplot as plt

#define data

x = np. array([50,33,110,45,66,87,42,37,73,95])
y = np. array ([100,62,180,70,84,93,50,52,90,120])

#find Line of best fat

a, b = np. polyfit(x, y, 1)

#add  points to plot

plt.scatter (x, y, color='purple')

#add

## add line of best fit to plot

plt.plot(x, a*x+b, color='steelblue', linestyle ='--', linewidth=2)
         
## add fittea regression equation to pLot

plt. text(30, 170, 'y = '+ '{:.2f}'.format(b) + ' + {:.2f} '.format(a) + 'x', size=14)

plt. show()