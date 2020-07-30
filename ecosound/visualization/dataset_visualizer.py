# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 12:45:13 2020

@author: xavier.mouy
"""

import hvplot.pandas
import param
import panel as pn

from bokeh.sampledata.iris import flowers

pn.extension()
inputs = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
    
class IrisDashboard(param.Parameterized):
    X_variable = param.Selector(inputs, default=inputs[0])
    Y_variable = param.Selector(inputs, default=inputs[1])
    
    @param.depends('X_variable', 'Y_variable')
    def plot(self):
        return flowers.hvplot.scatter(x=self.X_variable, y=self.Y_variable, by='species')
    
    def panel(self):
        return pn.Row(self.param, self.plot)

dashboard = IrisDashboard(name='Iris_Dashboard')

dashboard.panel().servable()