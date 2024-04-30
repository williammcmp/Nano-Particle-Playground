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
from src.LaserBeam import PulsedLaserBeam
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
st.markdown("Configure the setting of the pusle laser used for the laser ablation.")

slider_col, plot_col1, plot_col2 = st.columns([0.7, 1, 1])

with slider_col:
    options = ['PHAROS', 'SpitFire', 'Custom']
    laster_option = st.selectbox('Select the laser system.', options, index=0)
    # Laser input settings
    if laster_option == 'SpitFire':
    
        wavelength = 1064 * 1e-9 # default wavelength of 640 nm
        laser_power = 0.01 # default power of 1 W
        pulse_rate = 1 * 1e3 # default pulse rate MHz
        pulse_duration = 100 * 1e-13 # default pulse duration of 100 fs
        numerical_aperture = 0.10 # default NA of 0.1
        variables_dict = {
            'Wavelength (m)': f'{wavelength:.3g}',
            'Laser Power (W)': f'{laser_power:.3g}',
            'Pulse Rate (Hz)': f'{pulse_rate:.3g}',
            'Pulse Duration (s)': f'{pulse_duration:.3g}',
            'Numerical Aperture': f'{numerical_aperture:.3g}'
        }
        # Convert the dictionary into a DataFrame
        variables_df = pd.DataFrame(variables_dict, index=['Value']).T

        # Display the DataFrame in Streamlit
        st.dataframe(variables_df)

    elif laster_option == "PHAROS":
        wavelength = 1030 * 1e-9 # default wavelength of 640 nm
        laser_power = 7 # default power of 1 W
        pulse_rate = 200 * 1e3 # default pulse rate  kHz
        pulse_duration = 100 * 1e-13 # default pulse duration of 100 fs
        numerical_aperture = 0.14 # default NA of 0.1
        variables_dict = {
            'Wavelength (nm)': f'{wavelength*1e9:.4g}',
            'Laser Power (W)': f'{laser_power:.3g}',
            'Pulse Rate (KHz)': f'{pulse_rate*1e-3:.3g}',
            'Pulse Duration (fs)': f'{pulse_duration*1e13:.3g}',
            'Numerical Aperture': f'{numerical_aperture:.3g}'
        }
        # Convert the dictionary into a DataFrame
        variables_df = pd.DataFrame(variables_dict, index=['Value']).T

        # Display the DataFrame in Streamlit
        st.dataframe(variables_df)
        Beam = PulsedLaserBeam(wavelength, laser_power, pulse_rate, pulse_duration, beam_waist=15e-6)# Change the beam waist to be 15µm
    else:
        wavelength = st.number_input("Wavelength λ (nm)", min_value=300, max_value=1064, value=800) * 1e-9 # default wavelength of 640 nm
        laser_power = st.number_input("Laser power (W)", min_value=0.001, max_value=20.0, value=1.0) # default power of 1 W
        pulse_rate = st.number_input("Pulse rate (MHz)", min_value=0.01, max_value=1000.0, value=0.3) * 1e6 # default pulse rate of 80 MHz
        pulse_duration = st.number_input("Pulse Duration (fs)", min_value=1, max_value=500, value=100) * 1e-13 # default pulse duration of 100 fs
        numerical_aperture = st.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=2.0, value=0.14) # default NA of 0.1

        Beam = PulsedLaserBeam(wavelength, laser_power, pulse_rate, pulse_duration, numerical_aperture)


    z_air, z_silicon = Beam.calculate_rayleigh_range()

    # Intesnity abs
    z = np.linspace(0, z_silicon * 10, 100)
    I_gaus = (Beam.peak_intensity * 1e-4 / (1 + (z / z_silicon)**2))  # Intensity decay into the medium W/cm^
    I_k =  Beam.peak_intensity * np.exp(-Beam.calculate_absorption_coefficient() * z) # Intensity decay accounting for complex refractive index
    I_abs = I_gaus * (1 -  np.exp(-Beam.calculate_absorption_coefficient() * z)) # Intesnsity absorbed at each point 

    coulomb_limit = (465e3 * 2330) / (15.813 * 8.85e-12 * 377)
    threshold = I_k.max() * Beam.abs_threshold
    index = np.argmin(np.abs(I_k - threshold))
    z_abs_depth = z[index]
    

    # Sourced from FIG 3 -  L. Sudrie et. al 2002, "Femtosecond Laser-Induced Damage and Filamentary Propagation in Fused Silica"
    mulit_photon_ionisation_limit = 4e12 # (W/cm^2)

    st.dataframe(Beam.get_beam_statistics())

    



with plot_col1:
    # Parameters for the Gaussian distribution
    fig, ax = plt.subplots(figsize=(6, 4))

    # Generate x values
    x = np.linspace(-Beam.beam_waist - 0.002 * np.sqrt(Beam.beam_waist), Beam.beam_waist + 0.002 * np.sqrt(Beam.beam_waist), 1000)

    # Calculate the probability density function (PDF) for each x
    pdf = np.exp((-2 * x ** 2 ) / (Beam.beam_waist) ** 2)

    abs_factor = I_abs / Beam.intensity_per_pulse

    abs_profile = np.outer(pdf, abs_factor).T


    # Setup figure and gridspec
    fig = plt.figure(figsize=(10, 8))
    gs = plt.GridSpec(3, 1, height_ratios=[2, 2, 0.2], hspace=0)  # No gap between plots

    # Create subplots
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)

    # Plot Gaussian PDF on ax1
    ax1.plot(x, pdf, color='red')
    ax1.set_title("Gaussian Beam Intensity Profile")
    ax1.set_ylabel('Intensity (I / $I_0$)')
    
    ax1.axvline(x = Beam.beam_waist, color = "gray", linestyle='--' )
    ax1.axvline(x = -Beam.beam_waist, color = "gray", linestyle='--' )
    ax1.axvline(x = 0, color = "gray", linestyle='--' )
    ax1.axhline(y = 1 / np.e ** 2, color = "gray", linestyle='--' )

    ax.set_xlim([-Beam.beam_waist - 0.002 * np.sqrt(Beam.beam_waist), Beam.beam_waist + 0.002 * np.sqrt(Beam.beam_waist)])
    ax1.set_xticklabels([])  # Hide x-tick labels to avoid duplication

    # Plot Heatmap on ax2
    im = ax2.imshow(abs_profile, extent=[x.min(), x.max(), z.max(), z.min()], aspect='auto', origin='upper', cmap='inferno')
    ax2.set_xlabel('X position (m)')
    ax2.set_ylabel('Silicon Depth (m)')

    # Add color bar at the bottom
    cax = fig.add_subplot(gs[2])
    # cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    # cbar.set_label('Intensity (I / $I_0$)')

    st.pyplot()

    fig, ax = plt.subplots(figsize=(10,9))

    ax = PlotBeamFocal(ax, Beam.beam_waist, z_air, z_silicon, z_abs_depth)
    ax.axvline(x = 0, color = "gray", linestyle='--')
    ax.axhline(y = 0, color = "red")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z (m)")
    ax.set_title("Focual profile at interface")

    ax.set_xlim([-Beam.beam_waist - 0.002 * np.sqrt(Beam.beam_waist), Beam.beam_waist + 0.002 * np.sqrt(Beam.beam_waist)])
    ax.legend()
    st.pyplot(fig)



with plot_col2:

    # Intensity absorption profile along z-axis, (x,y = 0)
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(z, I_gaus, label="Gaussian Decay", alpha=0.7)
    ax.plot(z, I_k, label="Complex Decay")
    ax.plot(z, I_abs, label="Intensity absorbed", color='red', linestyle='--', alpha=0.7)
    ax.axvline(z_silicon, label="Silicon Rayleigh Range", color = "gray", linestyle='--', alpha=0.7)
    ax.axhline(threshold)
    ax.legend()
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Absrobed Intensity (w/cm^2)')
    ax.set_title("Silicon Absorption Profile")
    ax.set_ylim([0, I_gaus.max() * 1.1])

    st.pyplot()

    # energy abs : J (joules) = pi * w_0^2 * I * pulse duration / 2
    energy_abs = np.pi * (Beam.beam_waist ** 2) * I_abs * pulse_duration / 2 

    # jouels to eVV --> 1J = 6.242e18eV
    energy_abs_eV = (energy_abs * 6.242e18) 

    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(z, energy_abs_eV, label="Silicon") #me need to scale up or down on by 1e-4
    ax.axvline(z_silicon, label="Silicon Rayleligh Range", color = "gray", linestyle='--')
    # ax.axhline(coulomb_limit, label="Columb Limit", color='r')
    # ax.axhline(4.6, label="Silicon Work Function", color='r')

    ax.legend()
    ax.set_xlabel('z (m)')
    ax.set_ylabel('Absrobed Energy ( eV/cm^2 )')
    ax.set_title("Silicon Absorption Energy")

    st.pyplot()
# Plotting the energy absored


    # z = np.linspace(0, z_silicon * 1.3, 100)  # z-axis

    # s = np.tile(pdf, (z.size, 1))  # Duplicate the profile along the z-axis

    # fig, ax = plt.subplots(figsize=(7, 7))
    # c = ax.imshow(s, extent=[x.min(), x.max(), z.min(), z.max()], aspect='auto', origin='lower', cmap='inferno')
    # ax.set_xlabel('x (m)')
    # ax.set_ylabel('z (m)')
    # plt.colorbar(c, label='Intensity (normalized)')
    # st.pyplot(fig)





# ---------------
# Particle Settings
# ---------------
st.divider()
row0_1, row0_spacer2, row0_2, row0_spacer3 = st.columns((2, 1, 1.3, .1))
with row0_1:
    st.subheader('Particle Settings')
    st.markdown('Define the inital state of the paritcles from the ablation proces.')

    particle_options = ['Multi Photon Ionisation', 'Custom']

slider_col, plot_col1, plot_col2 = st.columns([1, 1, 1])

with slider_col:
    # particle_settings = st.selectbox('Particle Source Type', particle_options, index=0)

    # if particle_settings == 'Custom':
    particleNumber = st.number_input("Number of particles", min_value=10, max_value=1000, value=100, step=1)
    particleEnergy = st.number_input("Initial average particle energy (eV)", min_value=1, max_value=100, value=10, step=1) * 1e-16
    particleSize_min = st.number_input('Particle Size Minimum (nm)', min_value=10.0, max_value=150.0, value=10.0, step=1.0)
    particleSize_max = st.number_input('Particle Size Maximum (nm)', min_value=10.0, max_value=150.0, value=100.0, step=1.0)
    useNonConstantZ = st.checkbox("Use non-constant Z component", value=False)
    randomness = st.checkbox("Randomness 🤷", value=False)
    particleSize = (particleSize_min, particleSize_max)


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
        c.markdown("Define the Magnetic Field (T) Componets:")
        magneticX = c.number_input("Magnetic X", value=0.1, min_value = -0.2, max_value = 0.2)
        magneticY = c.number_input("Magnetic Y", value=0.0, min_value = -0.2, max_value = 0.2)
        magneticZ = c.number_input("Magnetic Z", value=0.0, min_value = -0.2, max_value = 0.2)

        st.markdown(f"Magnetic Field Magnitude: {np.linalg.norm(np.array([magneticX, magneticY, magneticZ]))} T")

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

        st.markdown(f"Electric Field Magnitude: {np.linalg.norm(np.array([electricX, electricY, electricZ]))} (V/m)")

        eleForce = Electric()
        # updateing the field -> obj will save the direction and magitude seperatlly
        eleForce.UpdateField(np.array([electricX, electricY, electricZ])) 
        simulation.AddForce([eleForce])

    # Ground Plane - could be an option for the simulation
    simulation.AddForce([GroundPlane()])

with plot_col1:
    fig, ax = simulation.PlotFroces()

    st.pyplot(fig)

st.divider()
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
        ax.set_xlim([-2e-5, 2e-5])
        ax.set_ylim([-2e-5, 2e-5])

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



