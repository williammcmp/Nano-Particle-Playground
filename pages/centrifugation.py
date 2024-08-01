# Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

def plot_centrifuge_pos(pos, size):
    fig, ax = plt.subplots(figsize=(5,4))
    x1, y1 = [0, 0.1], [0.01, 0.01]
    x2, y2 = [0, 0.1], [-0.01, -0.01]
    x3, y3 = [0, 0], [-0.01, 0.01]

    ax.scatter(pos, offset, s=size * 1e9, alpha=0.8)

    ax.plot(x1, y1, x2, y2, x3, y3, color='red', linewidth=2)
    ax.set_ylim([-0.011, 0.011])
    ax.set_xlim([-0.01, 0.11])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    st.pyplot(fig)

def plot_size_distro(pos, size):
    fig, ax = plt.subplots(figsize=(5,4))
    
    # Define the size ranges
    small_mask = size < 50 * 1e-9
    medium_mask = (size >= 50 * 1e-9) & (size < 150 * 1e-9)
    large_mask = (size >= 150 * 1e-9) & (size < 250 * 1e-9)
    very_large_mask = size > 250 * 1e-9

    # Plot density plots for each size category along positions
    sns.kdeplot(pos[small_mask], ax=ax, label='Small (0-50 nm)', color='blue', linewidth=2)
    sns.kdeplot(pos[medium_mask], ax=ax, label='Medium (50-150 nm)', color='green', linewidth=2)
    sns.kdeplot(pos[large_mask], ax=ax, label='Large (150-250 nm)', color='red', linewidth=2)
    sns.kdeplot(pos[very_large_mask], ax=ax, label='Very Large (>250 nm)', color='red', linewidth=2, linestyle='--')

    ax.set_title('Distribution of Particle Sizes Along Positions')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Density')
    ax.legend()

    st.pyplot(fig)

st.title("Centrifugation of SiNP Colloids")

slider_col, results_col = st.columns([1,3])

with slider_col:
    count = st.number_input("Number of Particles", 10, 1000, 10) # number of particles in the ellips
    angular_vel = st.number_input("Centrifuge speed (RPM)", 1, 40000, 4) * 2 * np.pi # RPM * 2pi of the centrifuge 
    duration = st.number_input("Duration (min)", 1, 120, 10) * 60 # Duration of Centrifugation

# Inital parms of colloids
position = np.random.uniform(0.0, 0.1, count)
# size = np.random.random_integers(5, 500, count) * 1e-9 # Radius of the particle
size = np.random.gamma(10,size=count) * 1e-8 # Radius of the particle
offset = np.random.uniform(-0.01, 0.01, count) # off sets in the y direction --> only modifying the x position

with slider_col:
    fig, ax = plt.subplots(figsize=(3,2))

    sns.kdeplot(size * 1e9, ax=ax, color='green', linewidth=2)
    
    ax.set_title('Distribution of Particle Sizes')
    ax.set_xlabel('Particle size (nm)')
    ax.set_ylabel('Density')
    st.pyplot(fig)

    st.divider()
    
    st.markdown(r"""
### Sedimentation Rate Equation

The sedimentation rate, \( \nu \), of a particle in a liquid is given by the equation:""")

    st.markdown(r''' $$\nu = \frac{2r_s^2(\rho_s - \rho_l)F}{9\eta}$$''')

    st.markdown(r'''
where:<br>
- \( r_s \) is the radius of the particle (sphere).<br>
- \( \rho_s \) is the density of the particle.<br>
- \( \rho_l \) is the density of the liquid.<br>
- \( F = \omega^2 r \), where \( \omega \) is the rotor speed (in rad/s) and \( r \) is the distance between the particle and the center of rotation (in cm).<br>
- \( \eta \) is the dynamic viscosity of the liquid.

This equation describes the sedimentation rate characteristic of a particle, which can be determined in an ultracentrifuge.
''')




# Defining values
rho_water, rho_silicon = 997, 2330
liquid_viscosity = 0.001002 # m kg/(m·s) | water at 20°C

sed_rate = (angular_vel ** 2) * 0.1 * ((2 * (size ** 2) * (rho_silicon - rho_water)) / (9 * liquid_viscosity)) # ⍵ * r (2r^2(ρ_s - ρ_w) / (p * liquid_viscosity) --> in cm/s
displacement = (sed_rate * duration * 1e-2)

new_pos = position - displacement
new_pos[new_pos < 0] = 0

with results_col:
    st.markdown("Results")
    before_col, after_col = st.columns([1,1])

    with before_col:
        st.text("Before Centriguation")
        plot_centrifuge_pos(position, size)
        plot_size_distro(position, size)

    with after_col:
        st.text("After Centriguation")
        plot_centrifuge_pos(new_pos, size)
        plot_size_distro(new_pos, size)


print(np.mean(displacement))