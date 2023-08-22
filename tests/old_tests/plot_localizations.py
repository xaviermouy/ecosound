from ecosound.core.measurement import Measurement
import os
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

in_dir =r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\Darienne_data\DangerRocksFiles-20230125T183142Z-001\results'
min_err_val = 0.2 # minimum localization uncertainty (in m)

# load data
meas = Measurement()
meas.from_netcdf(in_dir)

# filter based on uncertainty
meas.filter('x_err_span_m < @min_err_val',min_err_val=min_err_val,inplace=True)
meas.filter('y_err_span_m < @min_err_val',min_err_val=min_err_val,inplace=True)
meas.filter('z_err_span_m < @min_err_val',min_err_val=min_err_val,inplace=True)

# plot
X=meas.data.x_m
Y=meas.data.y_m
Z=meas.data.z_m


fig = plt.figure()
ax = fig.add_subplot(121,projection='3d')
ax.scatter(X, Y, Z, c='k',marker='.', alpha=0.5, s=5)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_aspect('equal')
ax.grid(True)
#ax.set_title('Localizations (uncertainty < '+str(min_err_val))

ax = fig.add_subplot(333)
ax.scatter(X, Y, c='k',marker='.', alpha=0.5, s=4)
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
#ax.set_aspect('equal', 'box')
#ax.grid(True)

ax = fig.add_subplot(336)
ax.scatter(X, Z, c='k',marker='.', alpha=0.5, s=4)
ax.set_xlabel('X (m)')
ax.set_ylabel('Z (m)')
#ax.set_aspect('equal','box')
#ax.grid(True)

ax = fig.add_subplot(339)
ax.scatter(Y, Z, c='k',marker='.', alpha=0.5, s=4)
ax.set_xlabel('Y (m)')
ax.set_ylabel('Z (m)')
#ax.set_aspect('equal','box')
#ax.grid(True)

fig.suptitle('Localizations (uncertainty < '+str(min_err_val)+'m)', fontsize=16)

plt.show()
print('done')

