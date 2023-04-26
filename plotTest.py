import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import cumtrapz

data = {
    'time': [0,1,2,3,4,5],
    'MP1':[5,9,8,7,6,4],
    'MP2':[8,6,2,4,8,9]
}

df = pd.DataFrame(data, dtype=float)

np_data = df['time'].to_numpy()
np_data = np_data[0:-1]
i = 10

for col in df.columns[1:]:
    v = cumtrapz(df[col], df['time'])
    np_data = np.column_stack((np_data, v*i))
    i = i * 10

df1 = pd.DataFrame(np_data, columns=['time','MP1','MP2'])
print(df1)

df.plot(kind='line', xlabel='Time', ylabel='Beschleunigung')
df1.plot()
plt.show()

