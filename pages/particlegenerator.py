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
from src.DataLoader import load_experimental_data
from src.streamlitText import *
from src.nanoParticlePlots import *

# Function to write data to a JSON file
def write_to_json(data, filename='output.json'):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file)

def pGen (n, mass, energy, reduceZ, randomness):
    p_positions = np.random.randn(n,2) # 1/3 is there to move the squish the distro between [-1,1]
    p_mass = np.random.uniform(mass[0] , mass[1], n)

    p_velocity = calVelocity(p_mass, p_positions, energy, reduceZ, randomness)
    
    zeros = np.zeros((p_positions.shape[0], 1)) + 0.0001
    p_positions = np.hstack((p_positions, zeros))
    
    dic = {
        "pos" : p_positions,
        "mass" : p_mass, 
        "vel": p_velocity
    }
    
    return dic
    

def calVelocity(mass, position, energy, reduceZ = False, randomness=False):

    velMag = np.sqrt(2 * (energy) / mass) # 1/3 needed to allow for energy across all axies
    
    
    if reduceZ:
        z = np.sqrt(9 - (position[:,0]**2 + position[:,1]**2)) # Reduces the velocity in the Z mag when further away from the origin (pre normalised z is always 1)
    else:
        z = np.ones((position.shape[0], 1)) # The virtical compoent of the vecotr before normalisation is always 1

    zMag = z.reshape(-1, 1) # Allows z postional values to be hstacked on the position array

    velDir = np.hstack((position, zMag)) 
    velNorm = velDir / np.linalg.norm(velDir, axis=1, keepdims=True) # ensure normalization along the correct axis
    Velocity = velMag.reshape(-1, 1) * velNorm

    if randomness:
        a = np.random.randn(position.shape[0],3) # Random offset in the particles inital velocity
        Velocity = Velocity + a # apply the random offset to the particles inital velocity
        Velocity[:, 2] = np.abs(Velocity[:, 2]) # makes Vz positive

    return Velocity


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
    particleEnergy = st.slider("Inital energy of particles", 1, 100, 10)
    particleMass = st.slider('Mass range of particles',0.0, 100.0, (25.0, 75.0))
    useNonConstantZ = st.checkbox("Use non-constant Z component", value=False)
    randomness = st.checkbox("Randomness 🤷", value=False)

    if st.button("Save inital distribution settings"):
        config_data = {
                "particleNumber": particleNumber,
                "particleEnergy": particleEnergy,
                "particleMass": particleMass,
                "useNonConstantZ": useNonConstantZ,
                "randomness": randomness
            }
        write_to_json(config_data)
        st.success("Configuration saved to output.json")

p = pGen(particleNumber, particleMass, particleEnergy, useNonConstantZ, randomness)


with plot_col1:
    fig, ax = plt.subplots()
    # Plot the norm alized histogram
    ax.hist(p['pos'][:,0], bins=30, density=True, color='green', alpha=0.7, label="X-distribution")
    ax.hist(p['pos'][:,1], bins=30, density=True, color='blue', alpha=0.7, label="Y-distribution")

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), color="red", alpha=0.5)


    # Set plot labels and legend
    plt.legend()
    plt.grid()
    plt.xlabel('Values')
    plt.ylabel('Density')
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
    plt.title('Inital position')
    
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
    ax.quiver(p['pos'][:,0], p['pos'][:,1], p['pos'][:,2], p['vel'][:,0], p['vel'][:,1], p['vel'][:,2], length=0.5, normalize=True, color='blue', arrow_length_ratio=0.2, alpha=0.1)

    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0,1])
    plt.title("Inital particle directions")

    st.pyplot(fig)


with col_plot2:
    x = x_data
    y = y_data
    velocities = p['vel']  # Replace with your actual velocities

    # Calculate the magnitude of velocities
    speeds = np.linalg.norm(velocities, axis=1)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create a 2D histogram (heatmap) of velocities using ax.hist2d
    h = ax.hist2d(x, y, bins=50, cmap='viridis', cmin=1, weights=speeds)

    # Add colorbar for reference
    cbar = fig.colorbar(h[3], ax=ax)
    cbar.set_label('Inital velocity')

    # Set axis labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Inital velocity (magnitude)')

    st.pyplot(fig)