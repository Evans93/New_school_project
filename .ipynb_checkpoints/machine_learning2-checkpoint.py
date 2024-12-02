import pandas as pd
import matplotlib.pyplot as plt

# Define data
data = {
    'Income': [2200, 3280, 3100, 2100, 2250, 3700, 3200, 3200, 2820, 3000, 3500],
    'Age': [18, 25, 22, 23, 23, 33, 26, 28, 29, 32],
    'Experience_Years': [3, 7, 5, 6, 7, 9, 5, 6, 8, 9]
}

df = pd.DataFrame(data)

# Scatter plot
plt.scatter(df['Income'], df['Age'], color='red')
plt.title('Income vs Age', fontsize=14)
plt.xlabel('Income', fontsize=14)
plt.ylabel('Age', fontsize=14)
plt.grid(True)
plt.show()
