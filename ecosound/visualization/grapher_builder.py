# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:02:51 2020

@author: xavier.mouy
"""


class BaseClass(object):
    """
    Base class for all graphers.

    All graphers need to inheritate from this BaseClass in order to be built
    by the GrapherFactory.
    """

    def __init__(self, grapher_name):
        self.grapher_name = grapher_name

    @classmethod
    def is_grapher_for(cls, grapher_name):
        """
        Check grapher name.

        Compare the requested grapher_name with each grapher class
        available.

        Parameters
        ----------
        grapher_name : str
        Name of the grapher class.

        Returns
        -------
        bool
        """
        return grapher_name == cls.__name__


def GrapherFactory(grapher_name, *args, **kwargs):
    """
    Grapher Factory.

    Loads the grapher class defined by grapher_name. Each grapher class must
    be added to the __init__.py file.

    For example :
    from .grapher1 import Grapher1

    where Grapher1 is the name of the grapher class to load from the .py
    file grapher1

    Parameters
    ----------
    grapher_name : str
        Name of the grapher class.
    *args : any
        Input arguments for the grapher selected.
    **kwargs : any
        Keyword arguments for the grapher selected.

    Raises
    ------
    ValueError
        If grapher_name doesn't correspond to any of the graphers available.

    Returns
    -------
    Grapher object

    """
    detec_list = []
    for cls in BaseClass.__subclasses__():
        detec_list.append(cls.__name__)
        if cls.is_grapher_for(grapher_name):
            return cls(grapher_name, *args, **kwargs)
    raise ValueError("Invalid grapher name. Grapher available: "
                     + str(detec_list))
