import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.Particle import Particle
from src.Simulation import Simulation
from src.Forces import *
from src.ParticleGenerator import *
from src.DataLoader import load_experimental_data
from src.streamlitText import *

# Load the experimnetal data and plot the overlay to the simualtion data
def ExperimentalMain(simulation):
    position, velocity, force, mass, charge = simulation.StreamletData()

    text_col, plot_col = st.columns([1, 2])

    with text_col:
        st.markdown(expermentalMainText())

        st.selectbox("Showing experimental data with", ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page"])

    with plot_col:
        data_df = load_experimental_data("NoBField")

        x = data_df["Width"]
        y = np.sqrt(data_df["X"]**2 + data_df["Y"]**2)


        fig, ax = plt.subplots()
        cmap = plt.get_cmap('viridis')
        normalize = plt.Normalize(charge.min(), charge.max())
        colors = cmap(normalize(charge))
        sa = ax.scatter(x,y, s=10)


        # Customize the plot (optional)
        ax.set_xlabel('Particle Diamater (nm)')
        ax.set_ylabel('Displacement from Origin (nm)')
        ax.set_title("Size Vs Displacement")
        ax.set_xlim(0,100)

        st.pyplot(fig)
    return True