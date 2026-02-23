# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:30:08 2020

@author: xavier.mouy
"""
import functools
import time


def listinput(func):
    """
    Decorator that wraps a single non-list argument in a list.

    Ensures that the first positional argument passed to the decorated function
    is always a list. If it is not already a list, it is wrapped in one before
    the function is called. Useful for functions that must accept both a single
    item and a list of items.
    """
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
    """
    Decorator that prints the execution time of the decorated function.

    Measures elapsed wall-clock time from just before the function is called
    to just after it returns, then prints the result to the console in seconds.
    """
    @functools.wraps(func)
    def wrapper_listinput(*args, **kwargs):
        tic = time.perf_counter()
        value = func(*args, **kwargs)
        toc = time.perf_counter()
        print(f"Executed in {toc - tic:0.4f} seconds")
        return value
    return wrapper_listinput
