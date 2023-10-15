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
from src.DataLoader import load_experimental_data
from src.streamlitText import *
from src.nanoParticlePlots import *

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
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=1, max_value=10000, value=3, step=500, disabled = True)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=10.0, step=0.5) / 100 # convert to seconds

    elif simMode == "Silicon Nano-Particles":
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=5, max_value=10000, value=100, step=100)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5, disabled = True)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5, disabled = True) / 100 # convert to seconds
    
    else:
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=1, max_value=10000, value=100, step=500)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=10.0, step=0.5) / 100 # convert to seconds

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

    elif simMode == "Silicon Nano-Particles":
        a = st.sidebar.expander("Nano-Particle Distribution Settings")
        positionType = "origin"
        positionX = 1
        positionY = 1
        positionZ = 0
        massRange = [0.1, 10] 
        AvgEnergy = 85
        charged = True

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

simMode = st.sidebar.selectbox("Simulation Mode:", ["Silicon Nano-Particles","Three Particle system (testing)", "Standard"])
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
        magneticX = c.number_input("Magnetic X", value=1.0)/1000
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
a = st.sidebar.expander("Simulation Constrains")

groundPlane = a.checkbox("Ground Plane", value=True)
if groundPlane:
    particleBounce = a.checkbox("Particle Bounce")
    if particleBounce:
        particleBounceFactor = a.number_input("Bounce factor (0 - no bounce, 1 lots of bounce)")
    else:
        particleBounceFactor = 0
rand = a.checkbox("Fixed Random Seed", value=True)
if rand:
    randSeed = a.number_input("Seed Number", step=1, value=2)
    random.seed(randSeed)

# ------------
# Settiing up the Sim
# ------------

# Generates the particles bases on what mode were are in
if simMode == "Silicon Nano-Particles":
    GenerateParticles(partilceNumber, simulation, mode, positionX, positionY, positionZ, massRange, avgEnergy, charged)
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
if groundPlane : simulation.AddConstraints([GroundPlane(particleBounceFactor)])


if simMode == "Silicon Nano-Particles":
    computeTime, numCals = simulation.Run(simDuration, simTimeStep)
else:
    computeTime, numCals = simulation.Run(simDuration, simTimeStep)


# ------------
# Introduction
# ------------

sim_info = f'''
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
    st.title('Characterisation of laser-ablated silicon nanoparticles')
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
row3_spacer1, row3_1, row3_spacer2 = st.columns((.1, 3, .1))
with row3_1:
    st.markdown(people_info())
    st.markdown("We invesigated how applying a magnetic field during the ablation process affects the particle's displacement from the ablation creator. If the particles are charged, then its expected the magnetic field would have an effect on the particle's displacment.")
    st.markdown("The source code can be fond on the [Nano Particle Playground GitHub repo](https://github.com/williammcmp/Nano-Particle-Playground)")



# ------------
# Displaying plots
# ------------
st.divider()

if simMode == "Silicon Nano-Particles":
    ExperimentalMain(simulation, sim_info)
else:
    position, velocity, force, mass, charge = simulation.StreamletData()

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
        sc = ax.scatter(mass, np.linalg.norm(position, axis=1),  c=colors, alpha=0.7)

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
        sc = ax.scatter(position[:, 0], position[:, 1], c=colors, alpha=0.7)

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

with st.expander("How to Use The Particle Simulation"):
    st.markdown(intro_info(simMode))

# with st.expander("Simulation Computation Info (Stats)"):
#     st.markdown(sim_info)
#     st.markdown("Froces:")
#     st.markdown(simulation.FroceList())