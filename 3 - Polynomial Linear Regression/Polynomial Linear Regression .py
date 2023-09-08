# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 16:52:52 2023

@author: 061885
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dt = pd.read_csv("polynomial+regression.csv", sep=";")

y = dt.araba_max_hiz.values.reshape(-1,1)
x = dt.araba_fiyat.values.reshape(-1,1)

plt.scatter(x,y)
plt.ylabel("Max-hiz")
plt.xlabel("Fiyat")
plt.show()

#%% polynomial linear regression

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree = 2) # x^2 icin deniyoruz

x_poly = poly_reg.fit_transform(x) # verdigimiz x inputunu fit eder ve ikinci dereceden poly feature çevirir.

lr = LinearRegression()
lr.fit(x_poly, y) # fit

#%% predict and visualize

y_head = lr.predict(x_poly)

plt.plot(x, y_head, color="red", label="poly - n^2")
plt.legend()
plt.show()
#%% if we use the linear model in this problem

lr2 = LinearRegression()
lr2.fit(x, y)

y2_head = lr2.predict(x)

plt.plot(x, y2_head, color="purple", label="linear")
plt.legend()
plt.show()
#%% bucak ibni sina no34 a sigorta halil aslantürk
poly_reg2 = PolynomialFeatures(degree = 4) # x^4 icin deniyoruz

x_poly2 = poly_reg2.fit_transform(x) # verdigimiz x inputunu fit eder ve ikinci dereceden poly feature çevirir.

lr3 = LinearRegression()
lr3.fit(x_poly2, y) # fit

y3_head = lr3.predict(x_poly2)

plt.plot(x, y3_head, color="orange", label="poly - n ^ 4")
plt.legend()
plt.show()

