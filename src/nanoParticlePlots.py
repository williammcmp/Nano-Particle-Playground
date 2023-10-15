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
        fig, ax = plotExperimentalData(dataSeries)

        ax.scatter(massToSize(mass), np.linalg.norm(position, axis=1), alpha=0.7)
        

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

    
        

    # Create a figure and axis
    fig, ax = plt.subplots()
    # ax.scatter(data['size'], data['displacement']) # raw data
    # Creates the error bard from experimental data
    for i in range(1,20):
        filted_data = data[(data['size'] >= (i * 5)-5) & (data['size'] <= (i * 5))] # grabs a range of sizes
        std_error = filted_data.std() # gets the standard error bars
        ax.errorbar(filted_data['size'].mean(), filted_data['displacement'].mean(), yerr=std_error['displacement'], fmt='o', capsize=5, color='r') # plots the avg displcement with error bars
    
    ax.set_xlabel('Particle size (nm)')
    ax.set_ylabel('Displacement (nm)')
    ax.set_title('Size Vs Displacement of the Silicon nano-partice')
    # ax.set_xlim(0, 100)
    ax.grid(True)

    return fig, ax

def massToSize(mass):
    density = 2330 # desnity of Silicon
    size = 2 * np.cbrt((mass * 3) / (4 * np.pi * density)) * 1e9
    return size