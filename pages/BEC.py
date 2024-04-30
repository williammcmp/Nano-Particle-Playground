# Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import math
import random

# Loading in the Simulation Objects
from src.Particle import Particle
from src.Simulation import Simulation
from src.Forces import *
from src.ParticleGenerator import *
from src.DataLoader import load_experimental_data
from src.streamlitText import *
from src.nanoParticlePlots import *


simulation = Simulation() # initalise the simulation object

# ------------
# Display properties
# ------------
# Set page layout to wide
st.set_page_config(layout="wide", 
                   page_title="Nano Particle Simulation", 
                   initial_sidebar_state="collapsed",
                   page_icon="img/NPP-icon-blueBG.png")
st.set_option('deprecation.showPyplotGlobalUse', False)


# makes the plots in line with the style of the application dark mode
rc = {'figure.figsize':(8,4.5),
        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 12,
        'axes.labelsize': 12,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12}
plt.rcParams.update(rc)

slider_col, plot_col1, plot_col2 = st.columns([0.7, 1, 1])

positions = []
velocity = []

with slider_col:
    st.markdown("A Classical Method")
    count = st.number_input("Number of Particles", 10, 1000, 100) # number of particles in the ellips
    time = st.number_input("Time of flight (s)", 1, 100, 1) # number of seconds the partiles are simulated for

while len(positions) < count:
    # Generate random point within bounding ellips
    x = np.random.uniform(low=-10, high=10)
    y = np.random.uniform(low=-7, high=7)
    v_x = np.random.normal(0, 1) 
    v_y = np.random.normal(0, 1) 
    
    # Check if point lies within the ellipsoid
    if (x**2 / 10**2) + (y**2 / 7**2) <= 1:
        positions.append([x, y, 0])
        velocity.append([v_x, v_y, 0])

particles = []
for row in range(count):
    particles.append(Particle(positions[row], velocity[row], mass = 1))

simulation = Simulation()
simulation.AddParticles(particles)

position, velocity, force, mass, charge = simulation.StreamletData()

# Inital positions
with plot_col1:
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(position[:, 0], position[:, 1], alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Inital position')
    ax.grid(True)
    ax.set_xlim(-15, 15)
    ax.set_ylim(-15, 15)

    st.pyplot(fig)

omputeTime, numCals = simulation.Run(time, time/100)

position, velocity, force, mass, charge = simulation.StreamletData()

# Inital positions
with plot_col2:
    fig, ax = plt.subplots(figsize=(10,6))

    sc = ax.scatter(position[:, 0], position[:, 1], alpha=0.5)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Final position')
    ax.grid(True)

    st.pyplot(fig)
