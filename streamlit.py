# particle_simulation_app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

from src.Particle import Particle
from src.Simulation import Simulation
from src.ParticleGenerator import *

# Define the Streamlit app
def main():
    # makes the plots in line with the style of the application
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
    # st.title("Particle Simulation App")

    # Create a simulation instance
    

    # # Display simulation info
    # if simulation.Particles:
    #     st.subheader("Simulation Info")
    #     st.write(f"Number of Particles: {len(simulation.Particles)}")
    #     st.write(f"Total Duration: {simulation.Duration} seconds")
    


    # Set page layout to wide
    st.set_page_config(layout="wide")

    # Sidebar
    st.sidebar.header("Simulation Settings")

    # Add sliders to the sidebar
    slider1 = st.sidebar.slider("Number of Particles", min_value=0, max_value=10000, value=50)
    slider2 = st.sidebar.slider("Simulation time (s)", min_value=0, max_value=30, value=5)
    slider3 = st.sidebar.slider("Time step (ms)", min_value=1, max_value=100, value=10)

    # Main content
    st.title("Particle Simulation App")

    simulation = Simulation()

    # Button to generate test particles
    # if st.button(f"Run Particle Simulation"):
    GenerateNanoParticles(slider1, simulation)
    simulation.Run(slider2, slider3/100)
    # st.success(f"Simculation Complete")


    fig, cbar = simulation.Plot()

    st.pyplot(fig)

    


        # You can add more features and visualization here

if __name__ == "__main__":
    main()
