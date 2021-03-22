# -*- coding: utf-8 -*-
"""
Created on Sun Mar  7 21:45:01 2021

@author: xavier.mouy
"""
import pandas as pd
import numpy as np; np.random.seed(0)
import seaborn as sns; sns.set_theme()
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

sns.set(rc={'axes.facecolor':'white', 'figure.facecolor':'white'})


#import altair as alt
infile = r'C:\Users\xavier.mouy\Documents\Reports_&_Papers\Papers\Kanes.etal_2021\GLMMdata_centeredSPL_AddedDatetimeColumn.csv'

# load and clean up data
data = pd.read_csv(infile)
data['date_time'] = pd.to_datetime(data['date_time'], format="'%d-%b-%Y %H:%M:%S'") # Convert date
data.drop(columns=['date','season','diel_per','SPL_100Hz_meanrem'], inplace=True)
data.set_index("date_time", inplace=True)

# add time and hour
data['Hour'] = pd.to_datetime(data.index).hour
#data['Day'] = data.index.map(lambda x: x.floor("d").tz_localize(tz=None))
data['Day'] = data.index.map(lambda x: x.strftime("%Y-%m-%d"))
#data['Day'] = data.index.map(lambda x: x.strftime("%d %b %Y"))


data_2D = data.reset_index(drop=True)
data_2D = data_2D.pivot_table(index='Hour', columns='Day',values='WSD_pres', aggfunc=np.max)
data_2D = data_2D.replace([np.inf, -np.inf,0,np.nan], np.nan)
fig, ax = plt.subplots(1)
ax = sns.heatmap(data_2D,
                 vmin=0,
                 vmax=1,
                 cmap='Greys',
                 cbar=False,
                 linewidth=0,
                 linecolor='k',
                 square=False,
                 )
ax.invert_yaxis()
plt.yticks(np.arange(0,25,2),np.arange(0,25,2), rotation=0, fontsize="12", va="center")
plt.xticks(fontsize="12")
ax.patch.set_edgecolor('black')
ax.patch.set_linewidth('2')
ax.set_ylabel('Hour of day')
ax.set_xlabel('')
fig.autofmt_xdate()
ax.grid(True, color='k', linestyle=':', linewidth=1)

