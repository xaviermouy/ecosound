from ecosound.core.measurement import Measurement
import os
import matplotlib.pyplot as plt

in_dir =r'J:\Taylor-Islet_LA_dep2\results'
meas = Measurement()
meas.from_netcdf(in_dir)

X=meas.data.x_m
Y=meas.data.y_m
Z=meas.data.z_m
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(X, Y, Z, marker=m)

print('done')
