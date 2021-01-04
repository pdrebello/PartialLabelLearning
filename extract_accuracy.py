#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 08:42:26 2020

@author: pratheek
"""

import numpy as np
import pandas as pd

datasets = ['MSRCv2','Yahoo! News','BirdSong','Soccer Player', 'Lost']
losses = ["cc_loss", "rl_loss", "naive_loss", "min_loss", "SelectR_sample_True", "SelectR_sample_False","SelectR_select_True","SelectR_select_False", ]

                
def get(filename):
    my_data = np.genfromtxt(filename, delimiter=',')
    #print(my_data)
    test_max = np.mean(my_data[290:,3])
    return test_max

k = 10

for filename in datasets:
    result = {}
    #print(filename)
    
    for loss in losses:
        try:
            for fold_no in range(k):
                result_filename = "results/"+filename+"/SelectR_"+str(loss)+"/results/"+str(fold_no)+"_out.txt"
                
                aggregate = 0
                with open(result_filename, "r") as f:
                    line = f.readline() 
                    line = f.readline() 
                    line = f.readline() 
                    aggregate += str(line)
            aggregate = aggregate/k
        except Exception as e:
            print(e)
            continue
            
        result[loss] = aggregate
    table = pd.DataFrame.from_dict(result)
    table.to_csv("results/"+filename+".csv")