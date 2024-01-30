#pages/particlegenerator.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy.stats as stats
import math
import json


from src.Particle import Particle
from src.Simulation import Simulation
from src.Forces import *
from src.ParticleGenerator import *
from src.DataLoader import *
from src.streamlitText import *
from src.nanoParticlePlots import *



def scale_convert(range, scaleFactor = 1e-9):
    """
    Convert the lower and upper values of a range from meters to nanometers.

    Parameters:
        rnage (tuple): A tuple containing the lower and upper values of the range in meters.
        scaleFactor (float, optional): A scaling factor the range tuple will be multplied by. Default is 10^-9, the scale factor of m -> nm

    Returns:
        tuple: A tuple containing the lower and upper values of the range in nanometers.

    Example:
    ```
    nm_scale = scale_convert([10, 100])
    ```
    """
    lowerRange = range[0] * scaleFactor
    upperRange = range[1] * scaleFactor

    return lowerRange, upperRange
    



row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.title('Characterisation of Laser-Ablated Silicon NanoParticles')
    st.subheader('Inital particle settings')
    st.markdown('Define the inital state of the paritcles from the ablation proces.')

with row0_2:
    image_container = st.container()

    # Add the image to the container
    image_container.image("img/swin_logo.png", width=200)

    # Apply CSS style to align the container to the right
    image_container.markdown(
        """
        <style>
        .st-dt {
            float: right;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

slider_col, plot_col1, plot_col2 = st.columns([1, 1, 1])

with slider_col:
    particleNumber = st.slider("Number of particles", 10, 10000, 1000)
    particleEnergy = st.slider("Inital energy of particles (J)", 1, 100, 10)
    particleSize = st.slider('Particle Size (nm)',10.0, 150.0, (10.0, 100.0))
    useNonConstantZ = st.checkbox("Use non-constant Z component", value=False)
    randomness = st.checkbox("Randomness 🤷", value=False)

    if st.button("Save inital distribution settings"):
        config_data = {
                "particleNumber": particleNumber,
                "particleEnergy": particleEnergy,
                "particleSize": scale_convert(particleSize),
                "useNonConstantZ": useNonConstantZ,
                "randomness": randomness
            }
        write_to_json(config_data)
        st.success("Configuration saved to output.json")

p = pGen(particleNumber, particleSize, particleEnergy, useNonConstantZ, randomness)


with plot_col1:
    fig, ax = plt.subplots()
    # Plot the norm alized histogram
    ax.hist(p['pos'][:,0], bins=30, density=False, color='green', alpha=0.6, label="X-distribution")
    ax.hist(p['pos'][:,1], bins=30, density=False, color='blue', alpha=0.6, label="Y-distribution")

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100) * 1e-6
    # ax.plot(x, stats.norm.pdf(x, mu, sigma), color="red", alpha=0.5)


    # Set plot labels and legend
    plt.legend()
    plt.grid()
    plt.xlabel('Positon (m)')
    plt.ylabel('Particle Count')
    plt.title('Normalized Axial Distribution of Particles')

    # Show the plot using Streamlit
    st.pyplot(fig)

with plot_col2:
    x_data = p['pos'][:,0]
    y_data = p['pos'][:,1]

    
    # Create a 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x_data, y_data, bins=50)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Plot the heatmap using imshow on the axes 'ax'
    image = ax.imshow(heatmap.T, extent=[xedges.min(), xedges.max(), yedges.min(), yedges.max()],
                    origin='lower', cmap='viridis')

    # Add colorbar
    cbar = plt.colorbar(image, ax=ax, label='Particle Density')

    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Inital particle position')
    
    st.pyplot(fig)


col_text, col_plot1, col_plot2 = st.columns([1, 1, 1])

with col_text:
    st.markdown('''
- Vector arrows are all equal in length.
- Particles further from the origin will experience a greater diffusion pressure.
- There is no random offset accounted for yet.''')


with col_plot1: 
    fig = plt.figure()  # Adjust the figsize as needed
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(p['pos'][:,0], p['pos'][:,1], p['pos'][:,2]*0, p['vel'][:,0], p['vel'][:,1], p['vel'][:,2], length=0.000005, normalize=True, color='blue', arrow_length_ratio=0.0002, alpha=0.1)



    radius = 2 * 1e-6
    center = (0, 0, 0)
    num_points = 100

    # Parametric equations for a circle in 3D space
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] + np.zeros_like(theta)  # All z-coordinates are zeros (lies in XY plane)

    ax.plot(x, y, z, color='r')

    ax.set_xlim([-3 * 1e-6, 3 * 1e-6])
    ax.set_ylim([-3 * 1e-6, 3 * 1e-6])
    ax.set_zlim([0,0.000005])
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    plt.title("Inital particle directions")

    st.pyplot(fig)


with col_plot2:
    x = x_data
    y = y_data
    velocities = p['vel'] # Replace with your actual velocities
    print(velocities)

    # Calculate the magnitude of velocities
    speeds = np.linalg.norm(velocities, axis=1)
    momentum = speeds * p['mass']
    print(np.min(speeds))
    print(np.max(speeds))
    

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a 2D histogram (heatmap) of velocities using ax.hist2d
    h = ax.hist2d(x, y, bins=50, weights=momentum)

    # Add colorbar for reference
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('momentum')

    # Set axis labels and title
    ax.set_xlabel('X-axis (m)')
    ax.set_ylabel('Y-axis (m)')
    ax.set_title('Inital momentum (magnitude) (m/s)')

    st.pyplot(fig)
    fig, ax = plt.subplots()

    ax.hist(speeds, bins=30, density=True, alpha=0.7)

    plt.xlabel('Particle Count')
    plt.ylabel('Speed (m/s)')
    plt.title('Distribution of inital particle speed')
    st.pyplot(fig)




