# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 16:08:18 2021

@author: xavier.mouy
"""
import matplotlib.pyplot as plt
import numpy as np

def func(f):
    x = np.linspace(-2, 2, 200)
    duration = 2
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x*2*np.pi*freq[90]))
    return mplfig_to_npimage(fig)

freq = np.linspace(0, 1, 100)

# creating animation
duration = 2
animation = VideoClip(func, duration = duration)

#animation.ipython_display(fps = 20, loop = True, autoplay = True)



 
# # importing movie py libraries
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage
 
# # numpy array
# x = np.linspace(-2, 2, 200)
 
# # duration of the video
# duration = 2
 
# # matplot subplot
# fig, ax = plt.subplots()
 
# # method to get frames
# def make_frame(t):
     
#     # clear
#     ax.clear()
     
#     # plotting line
#     ax.plot(x, np.sinc(x**2) + np.sin(x + 2 * np.pi / duration * t), lw = 3)
#     ax.set_ylim(-1.5, 2.5)
     
#     # returning mumpy image
#     return mplfig_to_npimage(fig)
 
# # creating animation
# animation = VideoClip(make_frame, duration = duration)
 
# # displaying animation with auto play and looping
# animation.ipython_display(fps = 20, loop = True, autoplay = True)