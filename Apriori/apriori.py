# -*- coding: utf-8 -*-
"""
Created on Wed May  6 14:53:03 2020

@author: Chinmay Kashikar
"""

#Apriori

#import library
import numpy as np   #cotain maths
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#import dataset
dataset=pd.read_csv('Market_Basket_Optimisation.csv',header=None)
transactions = []
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])

#Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#Visualising the result
results=list(rules)
results_list = []
for i in range(0, len(results)):
 results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t' + str(results[i][1]) + '\nInfo:\t' + str (results[i][2]))





