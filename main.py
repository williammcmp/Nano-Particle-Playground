import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# Importing all the required Classes
from Simulation import *
from Particle import *
from Forces import *

# Create a simulation
sim = Simulation()

# Create a particle and add it to the simulation
p1 = Particle([1, 0, 100], [1, 0, 0], 1)
sim.Particles.append(p1)

# Add gravity force to the simulation
sim.Forces.append(Gravity())

# Display information about the system
sim.Display()

# Update the simulation for a specified time interval (e.g., 3 seconds)
sim.Update(3)

# Display updated information
sim.Display()



# # Create a figure and axis
# fig, ax = plt.subplots()

# # Initialize an empty plot with a point marker
# point, = ax.plot([], [], 'bo')

# # Set the axis limits
# ax.set_xlim(0, 10)
# ax.set_ylim(0, 10)

# # Function to initialize the animation
# def init():
#     point.set_data([], [])
#     return point,

# # Function to update the animation at each frame
# def update(frame):
#     x = frame % 10
#     y = frame % 10
#     point.set_data(x, y)
#     return point,

# # Create the animation object
# ani = animation.FuncAnimation(fig, update, frames=range(100), init_func=init, blit=True)

# # To display the animation, you can save it to a file (e.g., 'animation.mp4') or display it in a Jupyter Notebook.
# # For Jupyter Notebook, you can use the following line to display the animation inline:
# # from IPython.display import HTML
# # HTML(ani.to_jshtml())

# # To save the animation as a video file, you can use the following line:
# # ani.save('animation.mp4', writer='ffmpeg')

# # To display the animation in a standalone window (not recommended for some environments):
# # plt.show()
