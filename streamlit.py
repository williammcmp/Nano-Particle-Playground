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
st.set_page_config(layout="wide", page_title="Nano Particle Simulation", initial_sidebar_state="collapsed")


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
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=1, max_value=1000, value=3, step=500, disabled = True)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5) / 100 # convert to seconds

    elif simMode == "Silicon Nano-Particles":
        # The max number of particles has been reduced in order to stop people form fucking crashing the server
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=5, max_value=1000, value=100, step=75, disabled = True)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=4, disabled = True)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5, disabled = True) / 100 # convert to seconds
    
    else:
        partilceNumber = st.sidebar.number_input("Number of Particles", min_value=1, max_value=10000, value=100, step=500)
        simDuration = st.sidebar.number_input("Simulation time (s)", min_value=0, max_value=30, value=5)
        simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5) / 100 # convert to seconds

    return partilceNumber, simDuration, simTimeStep 

def buildPartilceDistributions(simMode):
    if simMode == "Standard":
        a = st.sidebar.expander("Particle Distribution Settings")
        positionType = a.selectbox("Starting Position:", ["Origin", "Random", "off the wall"])
        if positionType == "Random ":
            positionX = a.number_input("Average inital X pos:")
            positionY = a.number_input("Average inital Y pos:")
            positionZ = a.number_input("Average inital Z pos:", min_value=0.0, value=1.0)
        elif positionType == "off the wall":
            positionX = a.number_input("Inital X pos:")
            positionY = a.number_input("Inital Y pos:")
            positionZ = a.number_input("Inital Z pos:", min_value=0.0, value=1.0)
        else:
            positionX = 1
            positionY = 1
            positionZ = 0

        massRange = a.slider('Range of Mass Particles (kg)', 0.0, 20.0, (1.0, 5.0))
        AvgEnergy = a.slider("Average Inital Energy (J)", value=3)
        charged = True
        chargedNev = a.checkbox("Enable Negative chages", value=True)
        chargedPos = a.checkbox("Enable Positive chages", value=True)

    elif simMode == "Silicon Nano-Particles":
        positionType = "origin"
        positionX = 1
        positionY = 1
        positionZ = 0
        massRange = [0.1, 10] 
        AvgEnergy = 85
        charged = True
        chargedNev = True
        chargedPos = True

    else:
        positionType = False
        positionX = 1
        positionY = 1
        positionZ = 0
        massRange = False
        AvgEnergy = False
        charged = True
        chargedNev = True
        chargedPos = True

    # Return the values as a tuple
    return positionType, positionX, positionY, positionZ, massRange, AvgEnergy, charged, chargedNev, chargedPos


# ------------
# Sidebar
# ------------
st.sidebar.header("Simulation Settings")
st.sidebar.markdown("Change the Simulation settings:  👇")

simMode = st.sidebar.selectbox("Simulation Mode:", ["Silicon Nano-Particles","Three Particle system (testing)", "Standard"])
st.sidebar.divider()

partilceNumber, simDuration, simTimeStep = buildSideBar(simMode)


st.sidebar.divider()
mode, positionX, positionY, positionZ, massRange, avgEnergy, charged, chargedNev, chargedPos= buildPartilceDistributions(simMode)




# Forces of the Simulation 
if simMode != "Silicon Nano-Particles":
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
            magneticX = c.number_input("Magnetic X", value=0.0)/1000
            magneticY = c.number_input("Magnetic Y", value=0.0)
            magneticZ = c.number_input("Magnetic Z", value=0.018) # default it is out of the page
        else:
            magneticX = np.array([0, 0, 0])
            magneticY = np.array([0, 0, 0])
            magneticZ = np.array([0, 0, 0])             

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

    wall = a.checkbox("Wall plane", value = False)

else: # condition for the Silicion Nano-Particle mode
    gravity = True
    electric = False
    magnetic = True
    magneticX = np.array([0, 0, 0])
    magneticY = np.array([0, 0, 0])
    magneticZ = np.array([0, 0, 0]) 
    groundPlane = True
    particleBounceFactor = 0
    wall = False


# ------------
# Settiing up the Sim
# ------------

# Generates the particles bases on what mode were are in
if simMode != "Three Particle system (testing)":
    GenerateParticles(partilceNumber, simulation, mode, positionX, positionY, positionZ, massRange, avgEnergy, charged,  chargedNev, chargedPos)
else:
    GenerateTestParticles(simulation)

# Allows for plotting inital positions
initalPos = simulation.Plot()

# Apply forces to the sim
if gravity :  simulation.AddForce([Gravity()])
if magnetic : simulation.AddForce([Lorentz(np.array([magneticX, magneticY, magneticZ]))])
if electric : simulation.AddForce([Lorentz(np.array([0, 0, 0]), np.array([electricX, electricY, electricZ]))])
if groundPlane : simulation.AddConstraints([GroundPlane(particleBounceFactor)])
if wall : simulation.AddConstraints([Wall()])

# ------------
# Introduction
# ------------

row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.title('Characterisation of Laser-Ablated Ailicon NanoParticles')
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
    with st.expander("About the Nano Particle Simulation"):
        st.markdown(sim_intro())
        st.image("img/SEM_image.png", use_column_width=True, caption="SEM of SiNPs. Positonal and size data was colleted from these images to from the experiemntal data in the model")




# ------------
# Displaying plots
# ------------
st.divider()

if simMode == "Silicon Nano-Particles":
    # ExperimentalMain(simulation, sim_info)
    
    # First Row of plots
    row1 = st.container()
    text_col, plot_col = row1.columns([1, 2])

    with text_col:
        st.markdown(expermentalMainText())

        dataSeries = st.selectbox("Select the direction of the magnetic field", ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page", "Magnetic Field Across the Page -Y", "Magnetic Field Across the Page +Y"])

        # Map data series option to a B field direction
        magneticDirection = {"No Magentic Field": np.array([0, 0, 0]),
                             "Magnetic Field out of the Page": np.array([0, 0, 0.18]),
                             "Magnetic Field into the Page": np.array([0, 0, -0.18]),
                             "Magnetic Field Across the Page -Y": np.array([0, -0.18, 0]),
                             "Magnetic Field Across the Page +Y": np.array([0, 0.18, 0])
                             }

        simulation.ChangeBField(magneticDirection[dataSeries])

        # fig, ax = plotExperimentalSummary()
        # st.pyplot(fig)

    # run the sim
    computeTime, numCals = simulation.NanoRun(simDuration, simTimeStep)

    # This needs to be here due to output of the sim run
    sim_info = f'''
    ```
    - Particles = {len(simulation.Particles):,}
    - Simulated time = {simDuration}s
    - Time Step intervals = {simTimeStep}s
    - Calacuation mode = {simMode}
    - Compute Time = {computeTime:.4}s
    ```
    '''

    
    # These variables makes it easer to make plots in streamlit
    position, velocity, force, mass, charge = simulation.StreamletData()

    with plot_col:
        fig, ax = plotExperimentalData(dataSeries)

        # There is some scaling on on the simulation results there.
        ax.scatter(mass*10, np.linalg.norm(position, axis=1) * 1e3, alpha=0.8, label="Simulation")

        # Add the 1/r^3 curve
        r = np.linspace(0.1, 10, 1000)  # Adjust the range as needed

        # Calculate the corresponding function values
        y = 1 / (r**3)

        ax.plot(r * 10, y * 1e4 + 2000, color="c", label=r"Expected $\frac{1}{r^3}$ Curve", linestyle='--', linewidth=3)

        # sets the legend's lables to be bright
        legend = ax.legend()
        for lh in legend.legendHandles:
            lh.set_alpha(1)
        st.pyplot(fig)

    col_1, col_2, col_3 = row1.columns([1,1,1])

    # Show the image of the magnetic field orentation
    with col_1:
        dataType = {
            "No Magentic Field" : "NoBField",
            "Magnetic Field out of the Page": "BFieldOut", 
            "Magnetic Field into the Page": "BFieldIn", 
            "Magnetic Field Across the Page -Y": "BFieldAcrossDown",
            "Magnetic Field Across the Page +Y": "BFieldAcrossUp"
        }
        if dataSeries != "No Magentic Field":
                
            st.image("img/magnetic_field_directions/" + dataType[dataSeries] + ".png", caption="Direction of the magnetic Field")


    # Show average displacement for the collected data series
    with col_2:
        fig, ax = plt.subplots(figsize=(10,7))
        fig, ax = plotExperimentalSummary(fig, ax)

        ax.scatter(0, 0, marker="o", s=2000, color='r', alpha=0.5) # origin of the plot
        ax.annotate("Ablation site", (0, 0), textcoords="offset points", xytext=(-60,10), ha='center', fontsize=12, color='red')
        ax.set_xlabel('X (μm)')
        ax.set_ylabel('Y (μm)')
        ax.set_title('Average SiNP displacement from ablation creator')
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    
    # show the particle distributions for the collected data series
    with col_3:
        fig, ax = plt.subplots(figsize=(10,7))
        fig, ax = plotExperimentalDistribution(dataSeries, fig, ax)

        ax.set_xlabel('particle diamater (nm)')
        ax.set_ylabel('Frequencey')
        ax.set_title(f'Size distributions of measured SiNPs from {dataSeries}')
        ax.legend()

        st.pyplot(fig)
        # fig, ax = plt.subplots(figsize=(10,7))
        # fig, ax = plotExperimentalDistributions(fig, ax)

        # ax.set_xlabel('particle diamater (nm)')
        # ax.set_ylabel('Frequencey')
        # ax.set_title('Size distributions of measured SiNPs for all data series')
        # ax.legend()

        # st.pyplot(fig)
    

    st.divider()
    row2 = st.container()
    # Second Row of plot - simulation figures
    text_col, spacer, plot_col, spacer2= row2.columns([2, 0.5, 2, 0.5])

    with text_col:
        # TODO make this cleaner
        st.markdown(simText())
        st.markdown(f'''**Simulation Stats:**''')
        st.markdown(sim_info)
        st.markdown(list_to_markdown_table(simulation.FroceList()))

    with plot_col:
        fig, ax = plotTrajectories(simulation, magneticDirection[dataSeries])

        st.pyplot(fig)

    col_1, col_2, col_3 = row2.columns([1,1,1])

    with col_1:
        # Plot the position of the simulated particles in a scatter plot
        fig, ax = plotSimulatedPosition(position, charge)
        fig, ax = plotExperimentalSummary(fig, ax)
        st.pyplot(fig)

    with col_2:
        fig, ax = plotSimulatedMassHistogram(mass)
        
        st.pyplot(fig)

    with col_3:
        fig, ax = plotMassDisplacement(position, charge)

        st.pyplot(fig)
else: 
    # Run the SIM for non Nano-partilce modes

    computeTime, numCals = simulation.Run(simDuration, simTimeStep)
    position, velocity, force, mass, charge = simulation.StreamletData()
    
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

    st.markdown(f"**Simulation Mode:** `{simMode}`")
    # First Row of plots
    row1 = st.container()
    plot_col1, plot_col2 = row1.columns([1, 1])
    
    with plot_col1:
        fig, ax = plotSimulatedPosition(position, charge)

        st.pyplot(fig)

    with plot_col2:
        fig, ax = plotTrajectories(simulation, np.array([magneticX, magneticY, magneticZ]))

        st.pyplot(fig)

    # fig, ax = plotForceTrajectories(simulation, np.array([magneticX, magneticY, magneticZ]))

    st.pyplot(fig)



with st.expander("How to Use The Particle Simulation"):
    st.markdown(how_to_use_info(simMode))

if simMode != "Silicon Nano-Particles":
    with st.expander("Simulation Computation Info (Stats)"):
        st.markdown(sim_info)
        st.markdown("Froces:")
        st.markdown(simulation.FroceList())

st.table(simulation.ParticleInfo())