
# import libraries
import pandas as pd
import matplotlib.pyplot as plt


# import data
df = pd.read_csv("linear_regression_dataset.csv",  sep=";")


# plot the data
plt.scatter(df.deneyim, df.maas)
plt.xlabel("deneyim")
plt.ylabel("maas")
plt.show()

#%%
from sklearn.linear_model import LinearRegression


linear_reg = LinearRegression()

x = df.deneyim.values.reshape(-1,1) # numpy ile calismak istedigimiz için numpy'a ceviriyoruz
y = df.maas.values.reshape(-1,1)

linear_reg.fit(x, y)

# b0' ı bulmak için (y eksenini kestigi nokta - intercept)
b0 = linear_reg.predict([[0]])

b0_ = linear_reg.intercept_

print("b0: ", b0)
print("b0_: ", b0_)

#b1'i bulmak için - slope

b1 = linear_reg.coef_

print("b1: ", b1) 

# yani buradan maas = 1663.89 + 1138.34 * deneyim olur

deneyim11 = b0 + b1 * 11
print("11 yillik deneyim:", deneyim11)

# bunu predict methodu ile de bulabiliriz

deneyim11_ = linear_reg.predict([[11]])

print("11 yillik deneyim - predict method: ", deneyim11_)

import numpy as np
arr = np.arange(0, 21).reshape(-1,1)

plt.scatter(x,y,color="blue")

plt.show()

y_head = linear_reg.predict(arr)

plt.plot(arr, y_head, color="red")


