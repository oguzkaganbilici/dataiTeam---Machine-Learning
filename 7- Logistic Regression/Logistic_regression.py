# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 12:00:56 2023

@author: 061885
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data.csv")

df.drop(["Unnamed: 32", "id"],axis=1, inplace=True)

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]
 
y = df.diagnosis.values
#x_data = df.iloc[:,1:]
x_data = df.drop(["diagnosis"], axis=1)

#%% normalization
xx = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
x = xx.values

#%% train - test split

from sklear.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

