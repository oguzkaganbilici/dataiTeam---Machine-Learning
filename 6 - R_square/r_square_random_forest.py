# -*- coding: utf-8 -*-
"""
Created on Sun Sep 10 13:35:46 2023

@author: oguzk
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("random+forest+regression+dataset.csv", sep=";", header=None)

x = df.iloc[:, 0].values.reshape(-1,1)
y = df.iloc[:, 1].values.reshape(-1,1)

#%% Random Forest

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators=100, random_state=42) 
#n_estimator = kac tane tree kullanacagımızı belirtiriz 
# random_state ile aynı sonuca ulasmak için bir sayı belirtiriz. farklı sayılar ile farklı random sonuclar cıkacaktır.

rf.fit(x,y)

x_ = np.arange(min(x), max(x), 0.03)

y_head = rf.predict(x_) # tahminler

#%% R-square

from sklearn.metrics import r2_score

print("R-score: ", r2_score(y, y_head))

