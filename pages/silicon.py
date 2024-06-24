# silicon.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

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
    
with st.expander("Laser Settins"):
    with slider_col:
    options = ['PHAROS', 'SpitFire', 'Custom']
    laster_option = st.selectbox('Select the laser system.', options, index=0, disabled=True)
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
        pulse_duration = 100 * 1e-15 # default pulse duration of 100 fs
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
        pulse_duration = st.number_input("Pulse Duration (fs)", min_value=1, max_value=500, value=100) * 1e-16 # default pulse duration of 100 fs
        numerical_aperture = st.number_input("Numerical Aperture (NA)", min_value=0.01, max_value=2.0, value=0.14) # default NA of 0.1

st.header("Heating Dynamics")
col1, spacer1, col2, spacer2= st.columns([1,0.3,1, 0.3])

with col1:
    st.markdown("There are multiple phases involved in the heating dynamics from ultra fast PLAL.")
    st.markdown("The Two Tempreature Model (TTM) is an analytical model that treats the electron gas and lattice as seperate systems to model how the overall tempreture of the system evolves over time.")

    st.latex(r'''c_e \frac{\partial T_e}{\partial t} = \text{div} \left( \kappa_e \nabla T_e \right) - \alpha \left( T_e - T_i \right) + Q''')
    st.latex(r'''c_i \frac{\partial T_i}{\partial t} = \text{div} \left( \kappa_i \nabla T_i \right) + \alpha \left( T_e - T_i \right)''')
with col2:
    st.image('img/heating-flow-chart.png', caption="Phases of a solid dilectric material heading from an ultrafast PLAL", use_column_width=True)



st.subheader("Two Temperature Model (TTM)")

st.subheader("Ionisation Regime")

st.subheader("Ablation Volumes")

st.header("Particle Ejection")

st.subheader("Particle Size Distribution")

st.header("References")
# st.markdown("[1] C. F. Bohren and D. R. Huffman. Absorption and scattering of light by small particles.John Wiley & Sons, 2008.")

