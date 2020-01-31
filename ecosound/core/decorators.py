# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 11:30:08 2020

@author: xavier.mouy
"""
import functools

def listinput(func):
    """Set input argument of function as a list if not already the case."""
    @functools.wraps(func)
    def wrapper_listinput(*args, **kwargs):
        #print(type(**kwargs))
        if type(*args) is not list:
            value = func([*args], **kwargs)
        else:
            value = func(*args, **kwargs)
        
        return value
    return wrapper_listinput