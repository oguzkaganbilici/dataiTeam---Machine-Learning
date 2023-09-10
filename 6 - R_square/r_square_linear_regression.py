# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 15:00:49 2023

@author: oguzk
"""

import pandas as pd

df  = pd.read_csv("linear_regression_dataset.csv", sep=";")

#%% linear regression

from sklearn.linear_model import LinearRegression

x = df.iloc[:, 0].values.reshape(-1,1)
y = df.iloc[:, 1].values.reshape(-1,1)

lr = LinearRegression()

lr.fit(x, y)

y_head = lr.predict(x)
#%% R-square
from sklearn.metrics import r2_score

print("r-score: ", r2_score(y, y_head))
