# -*- coding: utf-8 -*-
"""
Created on Wed May  6 23:44:26 2020

@author: Chinmay Kashikar
"""

#Thompson Sampling

#import library
import numpy as np   #cotain maths
import random
import matplotlib.pyplot as plt #help to plot nice chart.To plot something
import pandas as pd #to import dataset and to manage data set

#Importing Dataset
dataset=pd.read_csv('Ads_CTR_Optimisation.csv')

#Implementing Thompson Sampling
N = 10000 #no of users
d= 10 #no of adds
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0

for n in range(0,N):
    ad = 0
    max_random = 0
    for i in range(0,d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        
        if random_beta > max_random:
            max_random=random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad]=numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad]=numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    

#Visualising Result
plt.hist(ads_selected)    
plt.title('Histrogram Of ads Selection')
plt.xlabel('Ads')
plt.ylabel('Frequency of ads that selected')
plt.show()
