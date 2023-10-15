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
def ExperimentalMain(simulation, sim_info):
    position, velocity, force, mass, charge = simulation.StreamletData()
    
    # First Row of plots
    row1 = st.container()
    text_col, plot_col = row1.columns([1, 2])

    with text_col:
        st.markdown(expermentalMainText())

        dataSeries = st.selectbox("Showing experimental data with", ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page"])

    with plot_col:
        fig, ax = plotExperimentalData(dataSeries)

        # There is some scaling on on the simulation results there.
        ax.scatter(mass*10, np.linalg.norm(position, axis=1) * 1e3, alpha=0.7, label="Simulation")

        ax.legend()

        st.pyplot(fig)

    st.divider()
    row2 = st.container()
    # Second Row of plot - simulation figures
    text_col, spacer, plot_col, spacer2= row2.columns([2, 0.5, 2, 0.5])

    with text_col:
        st.markdown(simText())
        st.markdown(f'''**Simulation Stats:**''')
        st.markdown(sim_info)
        st.markdown(list_to_markdown_table(simulation.FroceList()))
        st.markdown("You have the flexibility to adjust simulation parameters, including the applied forces and other settings, through the side panel")

    with plot_col:
        fig = simulation.PlotPaths()
        st.pyplot(fig)

    col_1, col_2 = row2.columns([1,1])

    with col_1:
        # Reshape the position array to make it one-dimensional

        # Create the dataframe for easer ploting
        data = np.hstack((position, charge))

        fig, ax = plt.subplots(figsize=(10,6))

        for charge_val in [-1, 0, 1]:
            # Extract x and y positions of particles with charge value 0
            x_positions = data[data[:, 3] == charge_val][:, 0]
            y_positions = data[data[:, 3] == charge_val][:, 1]

            # Create a scatter plot
            ax.scatter(x_positions * 1e3, y_positions * 1e3, label=f"Charge {charge_val}", alpha=0.7)
        
        # Ploting the particles highlighed with different charges
        
        ax.set_xlabel('X (nm)')
        ax.set_ylabel('Y (nm)')
        ax.set_title('Simulated position of Silicion Nano-Particles')
        ax.grid(True)
        ax.legend()

        st.pyplot(fig)

    with col_2:
        fig, ax = plt.subplots(figsize=(10,6))
        ax.hist(np.linalg.norm(mass*10, axis=1), bins=10, edgecolor='k', alpha=0.7)

        ax.set_xlabel('Particle diamater (nm)')
        ax.set_ylabel('Frequency')
        ax.set_title("Histogram of Simulated Silicon nano-particle size")
        
        st.pyplot(fig)
        


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
    fig, ax = plt.subplots(figsize=(10,6))
    # ax.scatter(data['size'], data['displacement']) # raw data
    # Creates the error bard from experimental data
    for i in range(1,20):
        filted_data = data[(data['size'] >= (i * 5)-5) & (data['size'] <= (i * 5))] # grabs a range of sizes
        std_error = filted_data.std() # gets the standard error bars

        # Allows for label to be added to first plot (There could be a better solution)
        if i == 1:
            ax.errorbar(filted_data['size'].mean(), filted_data['displacement'].mean(), yerr=std_error['displacement'], fmt='o', capsize=5, color='r', label="Experimental") # plots the avg displcement with error bars
        else:
            ax.errorbar(filted_data['size'].mean(), filted_data['displacement'].mean(), yerr=std_error['displacement'], fmt='o', capsize=5, color='r') # plots the avg displcement with error bars

    
    ax.set_xlabel('Particle size (nm)')
    ax.set_ylabel('Displacement (nm)')
    ax.set_title('Size Vs Displacement of the Silicon nano-partice')
    # ax.set_xlim(0, 100)
    ax.set_ylim(0,10000)
    ax.grid(True)

    return fig, ax

def list_to_markdown_table(data):
    # Initialize the Markdown table
    fList = f'''
    **Forces Applied:**
    '''
    for row in data:
        fList += f'''
        - {row}
        '''
    

    return fList