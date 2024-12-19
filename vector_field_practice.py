import matplotlib.pyplot as plt
import numpy as np

# creating x, y coords over grid from 0 to 2pi with increments of 0.2 in both x & y dir.
X, Y = np.meshgrid(np.arange(0, 2 * np.pi, .2), np.arange(0, 2 * np.pi, .2))
# defines U & V arrays representing vector components
U = np.cos(X)
V = np.sin(Y)

# create figure & axis
fig1, ax1 = plt.subplots()
ax1.set_title('Arrows scale with plot width, not view')
# plots 2D vector field with:
#   X, Y coords of each vector 
#   U, V components (directions) of each vector
Q = ax1.quiver(X, Y, U, V, units='width')
# adds key to vector plot
#   0.9, 0.9 specifies position of key within figure
#   r'$2 \frac{m}{s}$' is label associated with each key
#   labelpos places label to EAST side of vector
#   coordinates = 'figure' is figure-relative positioning
qk = ax1.quiverkey(Q, 0.9, 0.9, 2, r'$2 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
plt.show()




# more spread out vector field
fig2, ax2 = plt.subplots()
ax2.set_title("pivot='mid'; every third arrow; units='inches'")
# plots 2D vector field with slices every 3rd point in both x & y
# directions to make a less dense grid
# place pivot in middle of each arrow
Q = ax2.quiver(X[::3, ::3], Y[::3, ::3], U[::3, ::3], V[::3, ::3],
               pivot='mid', units='inches')
qk = ax2.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
# plots scatter of red dots size 5
ax2.scatter(X[::3, ::3], Y[::3, ::3], color='r', s=5)
plt.show()




# color intensity of arrows and set arrow pivot
fig3, ax3 = plt.subplots()
ax3.set_title("pivot='tip'; scales with x view")
# calculates magnitude at each point, used to color the arrows through intensity
M = np.hypot(U, V)
# X, Y coords
# U, V x & y components
# units = 'x' scales arrows relative to x axis
# pivot = 'tip' positions base at tip of each arrow
# width=0.022 sets arrow width
# scale=1 / 0.15 scales the arrow length so that
#   vector magnitudes of approximately 0.15 units will be about one unit long in the plot.
Q = ax3.quiver(X, Y, U, V, M, units='x', pivot='tip', width=0.022,
               scale=1 / 0.15)
qk = ax3.quiverkey(Q, 0.9, 0.9, 1, r'$1 \frac{m}{s}$', labelpos='E',
                   coordinates='figure')
# color = '0.5' represents mid-gray, size = 1
ax3.scatter(X, Y, color='0.5', s=1)
plt.show()