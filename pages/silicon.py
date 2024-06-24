# silicon.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import NTMpy as ntm
import random
import numericalunits as u
u.reset_units('SI') # makes units easer to read and follow

from src.Particle import Particle
from src.Simulation import Simulation
from src.Forces import *
from src.ParticleGenerator import *
from src.utilities.DataLoader import load_experimental_data
from src.utilities.text.streamlitText import *
from src.utilities.plots.nanoParticlePlots import *

# ------------
# Display properties
# ------------
# Set page layout to wide
st.set_page_config(layout="wide", page_title="Nano Particle Simulation", initial_sidebar_state="collapsed")
st.set_option('deprecation.showPyplotGlobalUse', False)

# Streamlit theme settings
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
# Sidebar - simulation duration time
# ---------------
st.sidebar.markdown("Simulation Time Settings:")
simDuration = st.sidebar.number_input("Simulation time (s)", min_value=10, max_value=2000, value=250, step=50) / 1000 # convert to seconds
simTimeStep = st.sidebar.number_input("Time step (ms)", min_value=0.1, max_value=10.0, value=1.0, step=0.5) / 1000 # convert to seconds


col1, col2 = st.columns([2,1])
with col1:
    st.title('Laser Ablation of Silicon Nanoparticles')
    st.markdown("Dynamics of femotosecond Pulse Laser Ablation in Liquids (PLAL)")


with col2: 
    image_container = st.container()

    # Add the image to the container
    image_container.image("img/swin_logo.png", width=150)

    # Apply CSS style to align the container to the right
    image_container.markdown(
        """
        <style>
        .st-dt {
            float: right;
        }
        </style>
        """,
        unsafe_allow_html=True,)
    
with st.expander("Laser Settings"):
    options = ['PHAROS', 'SpitFire', 'Custom']
    col1, col2, col3 = st.columns([1,1,1])

    with col1:
        laser_option = st.selectbox('Select the laser system.', options, index=0, disabled=True)

    # Laser input settings
if laser_option == 'SpitFire':
    wavelength = 1064 * u.nm  # default wavelength of 1064 nm
    laser_power = 0.01 * u.W  # default power of 0.01 W
    pulse_rate = 1 * u.kHz  # default pulse rate of 1 kHz
    pulse_duration = 100 * u.fs  # default pulse duration of 100 fs
    numerical_aperture = 0.10  # default NA of 0.1

    variables_dict = {
        'Wavelength (m)': f'{wavelength:.3g}',
        'Laser Power (W)': f'{laser_power:.3g}',
        'Pulse Rate (Hz)': f'{pulse_rate:.3g}',
        'Pulse Duration (s)': f'{pulse_duration:.3g}',
        'Numerical Aperture': f'{numerical_aperture:.3g}'
    }

    # Convert the dictionary into a DataFrame
    variables_df = pd.DataFrame(variables_dict, index=['Value']).T


elif laser_option == "PHAROS":
    wavelength = 1030 * u.nm  # default wavelength of 1030 nm
    laser_power = 7 * u.W  # default power of 7 W
    pulse_rate = 200 * u.kHz  # default pulse rate of 200 kHz
    pulse_duration = 100 * u.fs  # default pulse duration of 100 fs
    numerical_aperture = 0.14  # default NA of 0.14

    variables_dict = {
        'Wavelength (nm)': f'{wavelength / u.nm:.4g}',
        'Laser Power (W)': f'{laser_power:.3g}',
        'Pulse Rate (kHz)': f'{pulse_rate / u.kHz:.3g}',
        'Pulse Duration (fs)': f'{pulse_duration / u.fs:.3g}',
        'Numerical Aperture': f'{numerical_aperture:.3g}'
    }

    # Convert the dictionary into a DataFrame
    variables_df = pd.DataFrame(variables_dict, index=['Value']).T

    Beam = PulsedLaserBeam(wavelength, laser_power, pulse_rate, pulse_duration, beam_waist=15 * u.um)  # Change the beam waist to be 15µm

else:
    wavelength = st.number_input("Wavelength λ (nm)", min_value=300, max_value=1064, value=800) * u.nm  # default wavelength of 800 nm
    laser_power = st.number_input("Laser power (W)", min_value=0.001, max_value=20.0, value=1.0) * u.W  # default power of 1 W
    pulse_rate = st.number_input("Pulse rate (MHz)", min_value=0.01, max_value=1000.0, value=0.3) * u.MHz  # default pulse rate of 0.3 MHz
    pulse_duration = st.number_input("Pulse Duration (fs)", min_value=1, max_value=500, value=100) * u.fs  # default pulse duration of 100 fs
    numerical_aperture = st.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=2.0, value=0.14)  # default NA of 0.14

    variables_dict = {
        'Wavelength (nm)': f'{wavelength / u.nm:.4g}',
        'Laser Power (W)': f'{laser_power:.3g}',
        'Pulse Rate (Hz)': f'{pulse_rate:.3g}',
        'Pulse Duration (s)': f'{pulse_duration:.3g}',
        'Numerical Aperture': f'{numerical_aperture:.3g}'
    }

    # Convert the dictionary into a DataFrame
    variables_df = pd.DataFrame(variables_dict, index=['Value']).T

    with col1:
        # Display the DataFrame in Streamlit
        st.dataframe(variables_df)

    with col2:
        st.markdown("add gaussian beam profile settings vis plots")

st.subheader("Two Temperature Model (TTM)")
# -----
# Setting up the TTM model
# -----



st.subheader("Ionisation Regime")

st.subheader("Ablation Volumes")

st.divider()
st.header("Particle Ejection")

st.subheader("Particle Size Distribution")

st.header("References")
# st.markdown("[1] C. F. Bohren and D. R. Huffman. Absorption and scattering of light by small particles.John Wiley & Sons, 2008.")

