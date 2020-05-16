# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:30:08 2020

@author: xavier.mouy
"""
import functools
import time


def listinput(func):
    """Set input argument of function as a list if not already the case."""
    @functools.wraps(func)
    def wrapper_listinput(*args, **kwargs):
        #  print(type(**kwargs))
        if type(*args) is not list:
            value = func([*args], **kwargs)
        else:
            value = func(*args, **kwargs)
        return value
    return wrapper_listinput


def timeit(func):
    """Print time it took to exceute function."""
    @functools.wraps(func)
    def wrapper_listinput(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Executed in {toc - tic:0.4f} seconds")
        return value
    return wrapper_listinput
