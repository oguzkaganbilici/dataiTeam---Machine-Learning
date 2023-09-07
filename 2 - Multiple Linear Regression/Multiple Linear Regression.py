# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:52:52 2023

@author: 061885
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


dt = pd.read_csv("multiple_linear_regression_dataset.csv", sep=";")

x = dt[["deneyim", "yas"]]
y = dt["maas"].values.reshape(-1,1)


multiple_linear_regression = LinearRegression()
multiple_linear_regression.fit(x, y)

print("b0: ", multiple_linear_regression.intercept_)
print("b1,b2:", multiple_linear_regression.coef_)

multiple_linear_regression.predict(np.array([[10, 35], [5, 35]]))

