# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
dt = pd.read_csv("decision+tree+regression+dataset.csv", sep=";", header=None)

x = dt.iloc[:,0].values.reshape(-1,1)
y = dt.iloc[:,1].values.reshape(-1,1)

#%% Decision Tree

from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor() # random_state = 0
tree_reg.fit(x, y)

x_ = np.arange(1, 10, 0.5).reshape(-1, 1)
y_head = tree_reg.predict(x_)



    #%% Visiualize

plt.scatter(x,y, color="red", label="real-values")
plt.plot(x_, y_head, color="green", label="Predicted values")
plt.xlabel("Tribune levels")
plt.ylabel("Price")
plt.show()