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

        dataSeries = st.selectbox("Showing experimental data with", ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page"])

    with plot_col:
        fig = plotExperimentalData(dataSeries)

        st.pyplot(fig)
    return True


def plotExperimentalData(dataSeries):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn"
    }
    
    # Loads the data frame of the specific data series selected by the user
    data_df = load_experimental_data(dataType[dataSeries])

    # Convert the X,Y positions to R or displcement values
    data = pd.DataFrame({"displacement":  np.sqrt(data_df["X"]**2 + data_df["Y"]**2),
                         "size": data_df["Width"]})
    
    grouped = data.groupby("size")
    avg_displacement = grouped['displacement'].mean()
    std_error = grouped['displacement'].std() / np.sqrt(grouped['displacement'].count())

    fig, ax = plt.subplots()

    # Plot the average sizes with error bars
    # ax.errorbar(avg_displacement.index, avg_displacement, yerr=std_error, fmt='o', capsize=5)
    ax.scatter(data['size'], data['displacement'])
    ax.set_xlabel('Particle size (nm)')
    ax.set_ylabel('Displacement (nm)')
    ax.set_title('size vs displacement')
    ax.set_xlim(0, 150)
    ax.grid(True)

    return fig