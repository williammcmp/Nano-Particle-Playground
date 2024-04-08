# ParticleSim.py  Streamlet App file

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

# ---------------
# Helper functions
# ---------------
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

def getDataSeries(simulation):
    if simulation.HasForce('Magnetic'):
        return "Magnetic Field out of the Page"
    else:
        return "No Magentic Field"

# ---------------
# Sidebar - simulation duration time
# ---------------
st.sidebar.markdown("Simulation Time Settings:")
simDuration = st.sidebar.number_input("Simulation time (s)", min_value=10, max_value=2000, value=250, step=50) / 1000 # convert to seconds
simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5) / 1000 # convert to seconds


# ---------------
# Page Header
# ---------------
row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.title('Nano-Particle Playground (NPP)')
with row0_2:
    image_container = st.container()

    # Add the image to the container
    image_container.image("img/NPP-icon-blueBG.png", width=150)

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


# ---------------
# Laser Settings
# ---------------
st.divider()
st.subheader("Laser Settings")
st.markdown("this is where the laser settings need to be placed")

# ---------------
# Particle Settings
# ---------------
st.divider()
row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.subheader('Particle Settings')
    st.markdown('Define the inital state of the paritcles from the ablation proces.')

slider_col, plot_col1, plot_col2 = st.columns([1, 1, 1])

with slider_col:
    particleNumber = st.slider("Number of particles", 10, 1000, 100)
    particleEnergy = st.slider("Inital average particle energy (eV)", 1, 100, 10) * 1e-16
    particleSize = st.slider('Particle Size (nm)',10.0, 150.0, (10.0, 100.0))
    useNonConstantZ = st.checkbox("Use non-constant Z component", value=False)
    randomness = st.checkbox("Randomness 🤷", value=False)

    # Automatically update the JSON file to read the settings
    # This could be removed and have the particle sim read directly from the
    # variables as needed
    config_settings = {
            "particleNumber": particleNumber,
            "particleEnergy": particleEnergy, 
            "particleSize": scale_convert(particleSize),
            "useNonConstantZ": useNonConstantZ,
            "randomness": randomness
        }
    write_to_json(config_settings)
    
    # Loads the particles from the JSON files
    particles = LoadParticleSettings()

    # adds the particles to the simulation obj
    simulation.AddParticles(particles)


p = pGen(particleNumber, particleSize, particleEnergy, useNonConstantZ, randomness)


with plot_col1:
    fig, ax = plt.subplots()
    # Plot the norm alized histogram
    ax.hist(p['pos'][:,0], bins=30, density=False, color='green', alpha=0.6, label="X-distribution")
    ax.hist(p['pos'][:,1], bins=30, density=False, color='blue', alpha=0.6, label="Y-distribution")

    # Set plot labels and legend
    plt.legend()
    plt.grid()
    plt.xlabel('Positon (m)')
    plt.ylabel('Particle Count')
    plt.title('Distribution of Particles')

    # Show the plot using Streamlit
    st.pyplot(fig)
    # TODO: Distribution of particle size
    # p_pd = pd.DataFrame.from_dict(p)
    # # small_p = p_pd[p_pd['mass'] < ]
    # fig, ax = plt.subplots()
    # # Plot the norm alized histogram
    # ax.hist(p['mass'], bins=30, density=False, color='green', alpha=0.6, label="X-distribution")
    # ax.hist(p['mass'][:,1], bins=30, density=False, color='blue', alpha=0.6, label="Y-distribution")


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

# ---------------
# Simulation Enviroment Setup
# ---------------
st.divider()

row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.subheader('Simulation Settings')
    st.markdown('Define the enviroment of the simulation')

slider_col, plot_col1 = st.columns([1, 1])

with slider_col:
    st.markdown("Forces")
    # Gravity
    if st.checkbox("Gravity", value=True):
        simulation.AddForce([Gravity()]) # Adds the ground plane to force list
    
    # Magnetic Force
    if st.checkbox("Magnetic field"):
        # TODO Make this better for non-side bar application
        c = st.container()
        c.markdown("Define the Magnetic Field (T):")
        magneticX = c.number_input("Magnetic X", value=0.1, min_value = -0.2, max_value = 0.2)
        magneticY = c.number_input("Magnetic Y", value=0.0, min_value = -0.2, max_value = 0.2)
        magneticZ = c.number_input("Magnetic Z", value=0.0, min_value = -0.2, max_value = 0.2)
        print(np.array([magneticX, magneticY, magneticZ]))

        magForce = Magnetic() # creating the Magnetic obj
        # updateing the field -> obj will save the direction and magitude seperatlly
        magForce.UpdateField(np.array([magneticX, magneticY, magneticZ])) 
        print(magForce.Field())
        simulation.AddForce([magForce]) # Adds mag force to force list 

    # Electric Force
    if st.checkbox("Electric field"):
        # TODO Make this better for non-side bar application
        c = st.container()
        c.markdown("Define the Electric Field (V/m):")
        electricX = c.number_input("Electric X", value=0.0)
        electricY = c.number_input("Electric Y", value=0.0)
        electricZ = c.number_input("Electric Z", value=0.0)

        eleForce = Electric()
        # updateing the field -> obj will save the direction and magitude seperatlly
        eleForce.UpdateField(np.array([electricX, electricY, electricZ])) 
        simulation.AddForce([eleForce])

    # Ground Plane - could be an option for the simulation
    simulation.AddForce([GroundPlane()])

with plot_col1:
    fig, ax = simulation.PlotFroces()

    st.pyplot(fig)

# ---------------
# Running the Simulation
# ---------------
if st.button("Run the Simulation"):
    computeTime, numCals = simulation.Run(simDuration, simTimeStep)
    position, velocity, force, mass, charge = simulation.StreamletData()

    sim_info = f'''
        ```
        - Particles = {len(simulation.Particles):,}
        - Simulated time = {simDuration}s
        - Time Step intervals = {simTimeStep}s
        - Compute Time = {computeTime:.4}s
        ```
        '''


    # ---------------
    # Plotting the Simulation results
    # ---------------
    st.divider()

    row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
    with row0_1:
        st.subheader('Simulation Figures')
        # st.markdown('Define the enviroment of the simulation')

    row1 = st.container()
    plot_col1, plot_col2 = row1.columns([1, 1])

    # Scatter plot of final postions of the particles
    with plot_col1:
        fig, ax = plotSimulatedPosition(position, charge)

        st.pyplot(fig)
        radius = np.sqrt(position[:,0] ** 2 + position[:,1] ** 2)
        inside = np.count_nonzero(radius < 1e-6)
        outside = np.count_nonzero(radius > 1e-6)
        stats = {'particle stats': [inside/len(mass), outside/len(mass)]}
        df = pd.DataFrame.from_dict(stats, orient='index')
        df.columns = ['inside', 'outside']
        st.table(df)
        st.caption("Proption of particles inside and outside the ablation site")

        st.markdown(f'''**Simulation Stats:**''')
        st.markdown(sim_info)

    # 3D plot of the particle trajectories
    with plot_col2:
        
        fig, ax = plotTrajectories(simulation)
        st.pyplot(fig)

    row2 = st.container()
    # Second Row of plot - simulation figures
    text_col, spacer, plot_col, spacer2= row2.columns([2, 0.5, 2, 0.5])

    # with text_col:
    #     st.markdown(f'''**Simulation Stats:**''')
    #     st.markdown(sim_info)

    # with plot_col:
    #     fig, ax = plotMassDisplacement(position, charge)

    #     st.pyplot(fig)



    # TODO: Work out what is happening with the plots of experimental and simulated data...
    # dataSeries = getDataSeries(simulation)

    # fig, ax = plotExperimentalData(dataSeries)

    # # There is some scaling on on the simulation results there.
    # ax.scatter(mass*10, np.linalg.norm(position, axis=1) * 1e3, alpha=0.8, label="Simulation")

    # # Add the 1/r^3 curve
    # r = np.linspace(0.1, 10, 1000)  # Adjust the range as needed

    # # Calculate the corresponding function values
    # y = 1 / (r**3)

    # ax.plot(r * 10, y * 1e4 + 2000, color="c", label=r"Expected $\frac{1}{r^3}$ Curve", linestyle='--', linewidth=3)

    # # sets the legend's lables to be bright
    # legend = ax.legend()
    # for lh in legend.legendHandles:
    #     lh.set_alpha(1)
    # st.pyplot(fig)


