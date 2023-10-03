# particle_simulation_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

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
        fastMode = st.sidebar.toggle("Enable Fast Mode") 
        partilceNumber = 3
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=1, max_value=100, value=10)/100 # convert to seconds
    else:
        fastMode = st.sidebar.toggle("Enable Fast Mode") 
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=0, max_value=10000, value=50)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=1, max_value=100, value=10)/100 # convert to seconds
    
    return fastMode, partilceNumber, simDuration, simTimeStep 


# # Display simulation info
# if simulation.Particles:
#     st.subheader("Simulation Info")
#     st.write(f"Number of Particles: {len(simulation.Particles)}")
#     st.write(f"Total Duration: {simulation.Duration} seconds")



# ------------
# Sidebar
# ------------
st.sidebar.header("Simulation Settings")
st.sidebar.markdown("Change the Simulation settings:  👇")

simMode = st.sidebar.selectbox("Simulation Mode:", ["Three Particle system (testing)", "Silicon Nano-Particles", "Standard"])

fastMode, partilceNumber, simDuration, simTimeStep = buildSideBar(simMode)

# Forces of the Simulation 
st.sidebar.divider()
st.sidebar.markdown("Forces selected:")
gravity = st.sidebar.checkbox("Gravity", value=True)
magnetic = st.sidebar.checkbox("Magnetic field")
if magnetic:
    c = st.sidebar.container()
    c.markdown("Define the Magnetic Field (T):")
    magneticX = c.number_input("Magnetic X", value=1.0)
    magneticY = c.number_input("Magnetic Y", value=0.0)
    magneticZ = c.number_input("Magnetic Z", value=0.0)

electric = st.sidebar.checkbox("Electric field")
if electric:
    c = st.sidebar.container()
    c.markdown("Define the Electric Field (T):")
    electricX = c.number_input("Electric X", value=0.0)
    electricY = c.number_input("Electric Y", value=0.0)
    electricZ = c.number_input("Electric Z", value=0.0)

# Constraints of the Simulation
st.sidebar.divider()
st.sidebar.markdown("Constrains selected:")
if fastMode :
    groundPlane = st.sidebar.checkbox("Ground Plane", value=False, disabled=True)
else:
    groundPlane = st.sidebar.checkbox("Ground Plane", value=True)
    if groundPlane:
        particleBounce = st.sidebar.checkbox("Particle Bounce")
        if particleBounce:
            particleBounceFactor = st.sidebar.number_input("Damping coeffiecent")



# ------------
# Introduction
# ------------
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
    st.markdown(f"**Simulation Mode:** `{simMode}`")
    



# ------------
# Settiing up the Sim
# ------------

# Generates the particles bases on what mode were are in
if simMode == "Silicon Nano-Particles":
    GenerateNanoParticles(partilceNumber, simulation)
elif simMode == "Standard":
    GenerateParticles(partilceNumber, simulation)
else:
    GenerateTestParticles(simulation)

initalPos = simulation.Plot()

# Apply forces to the sim
if gravity :  simulation.AddForce([Gravity()])
if magnetic : simulation.AddForce([Lorentz(np.array([magneticX, magneticY, magneticZ]))])
if electric : simulation.AddForce([Lorentz(np.array([0, 0, 0]), np.array([electricX, electricY, electricZ]))])
if groundPlane: simulation.AddConstraints([GroundPlane()])

# Change the run mode
if fastMode:
    computeTime, numCals = simulation.FastRun(simDuration, simTimeStep)
else:
    computeTime, numCals = simulation.Run(simDuration, simTimeStep)


# ------------
# Sim Info
# ------------

with st.expander("See Simulation Info"):
    info1_spacer, info1, info2, info3 = st.columns([0.1, 1,1, 1])

    info1.write(f'''```
    - Particles = {len(simulation.Particles)}
- Simulated time = {simDuration}s
- Time intervals = {simTimeStep}s
- Compute Time = {computeTime:.4}s
- Total number of calculations = {numCals}''')


    info2.markdown(f"**Forces:**")
    for force in simulation.Forces:
        info2.text(f"{force.Info()}")

    # info3.markdown(f"**Constraints:**")
    # for constaint in simulation.Constraints:
    #     info3.text(f"{constaint.Info()}")



# ------------
# Displaying plots
# ------------

scatter = st.container()

spacer_1, graphs1, spacer_2, graphs2, spacer_3 = scatter.columns([0.1, 3, 0.1, 3, 0.1])

with graphs1:
    st.markdown("Inital positon")
    st.pyplot(initalPos)

with graphs2:
    st.markdown("Final position")
    st.pyplot(simulation.Plot())


other = st.container()
other.divider()

spacer_1, graphs1, spacer_2, graphs2, spacer_3 = other.columns([0.1, 3, 0.1, 3, 0.1])

with graphs1:
    st.pyplot(simulation.Histogram())

with graphs2:
    if not fastMode:
        fig = simulation.PlotPaths()
        st.pyplot(fig)









