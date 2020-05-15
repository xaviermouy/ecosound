# -*- coding: utf-8 -*-
"""
Created on Wed May 13 13:44:29 2020

@author: xavier.mouy
"""
from time import sleep, perf_counter
from dask import delayed, compute, visualize

# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster()
# client = Client(cluster,processes=False)


def inc(x):
    sleep(1)
    return x + 1

def add(x, y):
    sleep(1)
    return x + y

tic = perf_counter()

data = range(1,80,1)
# Sequential code
results = []
for x in data:
    y = delayed(inc)(x)
    results.append(y)
    
total = delayed(sum)(results)
C=total.compute()
#total.visualize('dsds.png')

toc = perf_counter()
print(f"Dexecuted in {toc - tic:0.4f} seconds")