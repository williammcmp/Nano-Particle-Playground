# particle_simulation_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from src.Particle import Particle
from src.Simulation import Simulation
from src.Forces import *
from src.ParticleGenerator import *

simulation = Simulation() # initalise the simulation object

# ------------
# Display properties
# ------------
# Set page layout to wide
st.set_page_config(layout="wide", page_title="Particle Playground")


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

# ------------
# Helper functions
# ------------

def buildSideBar(simMode):
    if simMode == "Three Particle system (testing)":
        partilceNumber = 3
    else:
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=50, max_value=10000, value=100, step=50)
     
    simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
    simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=10.0, step=0.5)/100 # convert to seconds

        
  
    return partilceNumber, simDuration, simTimeStep 

def buildPartilceDistributions(simMode):
    if simMode == "Standard":
        a = st.sidebar.expander("Particle Distribution Settings")
        positionType = a.selectbox("Starting Position:", ["Origin", "Random"])
        if positionType == "Random":
            positionX = a.number_input("Average inital X pos:")
            positionY = a.number_input("Average inital Y pos:")
            positionZ = a.number_input("Average inital Z pos:", min_value=0, value=1)
        else:
            positionX = 1
            positionY = 1
            positionZ = 0

        massRange = a.slider('Range of Mass Particles (kg)', 0.0, 20.0, (1.0, 5.0))
        AvgEnergy = a.slider("Average Inital Energy (J)", value=3)
        charged = a.checkbox("Charged Particles (+, 0, -)", value=True)
    else:
        positionType = False
        positionX = 1
        positionY = 1
        positionZ = 0
        massRange = False
        AvgEnergy = False
        charged = True

    # Return the values as a tuple
    return positionType, positionX, positionY, positionZ, massRange, AvgEnergy, charged


# ------------
# Sidebar
# ------------
st.sidebar.header("Simulation Settings")
st.sidebar.markdown("Change the Simulation settings:  👇")

simMode = st.sidebar.selectbox("Simulation Mode:", ["Standard","Three Particle system (testing)", "Silicon Nano-Particles"])
st.sidebar.divider()

partilceNumber, simDuration, simTimeStep = buildSideBar(simMode)


st.sidebar.divider()
mode, positionX, positionY, positionZ, massRange, avgEnergy, charged = buildPartilceDistributions(simMode)




# Forces of the Simulation 
# st.sidebar.divider()
a = st.sidebar.expander("Simulation Forces")
gravity = a.checkbox("Gravity", value=True)

# Disabled charged based forces if particles are not charged
if not charged: 
    electric = a.checkbox("Electric field", disabled=True, value=False)
    magnetic = a.checkbox("Magnetic field", disabled=True, value=False)
else:
    magnetic = a.checkbox("Magnetic field")
    if magnetic:
        c = a.container()
        c.markdown("Define the Magnetic Field (T):")
        magneticX = c.number_input("Magnetic X", value=1.0)
        magneticY = c.number_input("Magnetic Y", value=0.0)
        magneticZ = c.number_input("Magnetic Z", value=0.0)

    electric = a.checkbox("Electric field")
    if electric:
        c = a.container()
        c.markdown("Define the Electric Field (T):")
        electricX = c.number_input("Electric X", value=0.0)
        electricY = c.number_input("Electric Y", value=0.0)
        electricZ = c.number_input("Electric Z", value=0.0)

# Constraints of the Simulation
# st.sidebar.divider()
a = st.sidebar.expander("Simulation Constrains")

groundPlane = a.checkbox("Ground Plane", value=True)
if groundPlane:
    particleBounce = a.checkbox("Particle Bounce")
    if particleBounce:
        particleBounceFactor = a.number_input("Damping coeffiecent")
rand = a.checkbox("Fixed Random Seed", value=True)
if rand:
    randSeed = a.number_input("Seed Number", step=1, value=2)
    random.seed(randSeed)

# ------------
# Settiing up the Sim
# ------------

# Generates the particles bases on what mode were are in
if simMode == "Silicon Nano-Particles":
    GenerateNanoParticles(partilceNumber, simulation)
elif simMode == "Standard":
    GenerateParticles(partilceNumber, simulation, mode, positionX, positionY, positionZ, massRange, avgEnergy, charged)
else:
    GenerateTestParticles(simulation)

# Allows for plotting inital positions
initalPos = simulation.Plot()

# Apply forces to the sim
if gravity :  simulation.AddForce([Gravity()])
if magnetic : simulation.AddForce([Lorentz(np.array([magneticX, magneticY, magneticZ]))])
if electric : simulation.AddForce([Lorentz(np.array([0, 0, 0]), np.array([electricX, electricY, electricZ]))])
if groundPlane : simulation.AddConstraints([GroundPlane()])

# Run the Sim
computeTime, numCals = simulation.Run(simDuration, simTimeStep)

# ------------
# Introduction
# ------------

intro_info = f'''

1. Choose the simulation mode from the sidebar. Current mode `{simMode}`
2. Adjust the simulation settings in the sidebar as needed.
3. The simulation will update after each setting change.
4. Explore the results and visualizations on the main page.

## Simulation Modes

- **Standard:** Have complete control over the simulation, particles and forces.
- **Three Particle system:** This mode demonstrates the behavior of three particles.
- **Silicon Nano-Particles:** (WIP) Simulates silicon nanoparticles in a magnetic field.

## Particle Initial Distribution (normal)

- Choose whether particles start at the origin or have random positions.
- Specify the average initial positions (X, Y, Z).
- Adjust the average mass and inital Kenetic Energy.
- Toggle charged particles (positive, negative, or neutral).

All distribution follow a normal distribution.

Feel free to explore and experiment with different settings to see how the particles behave!
'''

sim_info = f'''
## General Info:

```
- Particles = {len(simulation.Particles):,}
- Simulated time = {simDuration}s
- Time Step intervals = {simTimeStep}s
- Calacuation mode = {simMode}
- Compute Time = {computeTime:.4}s
- Total number of calculations = {numCals:,}
```
'''

row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.title('Particle Playground - Beta')
with row0_2:
    st.text("")
    st.subheader('Developed by [William McMahon-Puce](https://www.linkedin.com/in/william-mcmahon-puce-b9b3a9210//)')
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3, .1))
with row3_1:
    st.markdown("Hi there, this Streamlit app serves as a sandbox for simulating and experimenting with the behavior, interactions and dynamics of partiicles. Different simulation modes will allow you to play with sized regimes of particles.")
    st.markdown("You can find the source code in the [Nano Particle Playground](https://github.com/williammcmp/Nano-Particle-Playground)")
    
    with st.expander("How to Use The Particle Simulation"):
        st.markdown(intro_info)
    
    with st.expander("Simulation Computation Info (Stats)"):
        st.markdown(sim_info)


# ------------
# Displaying plots
# ------------

position, velocity, force, mass, charge = simulation.StreamletData()

st.divider()
st.markdown(f"**Simulation Mode:** `{simMode}`")

scatter = st.container()

spacer_1, graphs1, spacer_2, graphs2, spacer_3 = scatter.columns([0.1, 3, 0.1, 3, 0.1])

with graphs1:
    st.markdown("Inital positon")
    st.pyplot(initalPos)

with graphs2:
    st.markdown("Final position")
    st.pyplot(simulation.Plot())


other = st.container()
# other.divider()

spacer_1, graphs1, spacer_2, graphs2, spacer_3 = other.columns([0.1, 3, 0.1, 3, 0.1])

with graphs1:
    st.pyplot(simulation.Histogram())

    fig, ax = plt.subplots()
    ax.hist(np.linalg.norm(mass, axis=1), bins=10, edgecolor='k', alpha=0.7, color="#1f7c61")

    # Customize the plot (optional)
    ax.set_xlabel('Masses (kg)')
    ax.set_ylabel('Frequency')
    ax.set_title("Mass of particles")

    st.pyplot(fig)

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    normalize = plt.Normalize(charge.min(), charge.max())
    colors = cmap(normalize(charge))
    sc = ax.scatter(mass, np.linalg.norm(position, axis=1),  c=colors, cmap=cmap, alpha=0.7)

    # Add a colorbar to indicate charge values
    cbar = plt.colorbar(sc, ax=ax, label='Charge')

    # Customize the plot (optional)
    ax.set_xlabel('Mass of Particle (kg)')
    ax.set_ylabel('Displacement from Origin (m)')
    ax.set_title("Mass Vs Displacement")

    st.pyplot(fig)

with graphs2:
    # Create a colormap for charge values
    cmap = plt.get_cmap('viridis')
    normalize = plt.Normalize(charge.min(), charge.max())
    colors = cmap(normalize(charge))

    # Create a scatter plot with colored points
    fig, ax = plt.subplots()
    sc = ax.scatter(position[:, 0], position[:, 1], c=colors, cmap=cmap, alpha=0.5)

    # Add a colorbar to indicate charge values
    cbar = plt.colorbar(sc, ax=ax, label='Charge')

    # Customize the plot (optional)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Particle Scatter Plot')

    # Display the plot in Streamlit
    st.pyplot(fig)

    fig = simulation.PlotPaths()
    st.pyplot(fig)









