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


def plotExperimentalSummary(fig, ax):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn", 
        "Magnetic Field Across the Page": "BFieldAcross"
    }

    dataSeries = ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page", "Magnetic Field Across the Page"]

    for series in dataSeries:
        data = load_experimental_data(dataType[series])

        ax.scatter(data['X'].mean() * 1e-3, data['Y'].mean() * 1e-3, label=f"Avg pos - {series}", marker='1', s = 350)

    ax.legend()

    return fig, ax
        

def plotExperimentalData(dataSeries):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn", 
        "Magnetic Field Across the Page": "BFieldAcross"
    }
    
    # Loads the data frame of the specific data series selected by the user
    data_df = load_experimental_data(dataType[dataSeries])

    # Convert the X,Y positions to R or displcement values
    data = pd.DataFrame({"displacement":  np.sqrt(data_df["X"]**2 + data_df["Y"]**2),
                         "size": data_df["Width"]})

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10,6))
    ax.scatter(data['size'], data['displacement'], c='g', alpha=0.05, label="Experimental - Raw") # raw data
    # Creates the error bard from experimental data
    for i in range(1,20):
        filted_data = data[(data['size'] >= (i * 5)-5) & (data['size'] <= (i * 5))] # grabs a range of sizes
        std_error = filted_data.std() # gets the standard error bars

        # Allows for label to be added to first plot (There could be a better solution)
        if i == 1:
            ax.errorbar(filted_data['size'].mean(), filted_data['displacement'].mean(), yerr=std_error['displacement'], fmt='o', capsize=5, color='r', label="Experimental - Averages") # plots the avg displcement with error bars
        else:
            ax.errorbar(filted_data['size'].mean(), filted_data['displacement'].mean(), yerr=std_error['displacement'], fmt='o', capsize=5, color='r') # plots the avg displcement with error bars

    
    ax.set_xlabel('Particle size (nm)')
    ax.set_ylabel('Displacement (nm)')
    ax.set_title(f'Silicon Nano-Particles Size Vs Displacement - {dataSeries}')
    ax.set_xlim(0, 100)
    ax.set_ylim(0,14000)
    ax.grid(True)

    return fig, ax

def plotSimulatedPosition(position, charge):

    # Create a scatter plot with colored points
    fig, ax = plt.subplots(figsize=(10,6))
    sc = ax.scatter(position[:, 0], position[:, 1], c=charge, alpha=0.5)

    # Add a colorbar to indicate charge values
    cbar = plt.colorbar(sc, ax=ax, label='Charge')

    # Customize the plot (optional)
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_title('Simulated position of Silicion Nano-Particles')
    ax.grid(True)

    return fig, ax

def plotSimulatedMassHistogram(mass):
    fig, ax = plt.subplots(figsize=(10,7))
    ax.hist(np.linalg.norm(mass*10, axis=1), bins=10, edgecolor='k', alpha=0.7)

    ax.set_xlabel('Particle diamater (nm)')
    ax.set_ylabel('Frequency')
    ax.set_title("Simulated Silicon nano-particle size")

    return fig, ax


def plotTrajectories(simulation, direction):

   
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')

    for particle in simulation.Particles:
        ax.plot(particle.History[:, 0], particle.History[:, 1], particle.History[:, 2]) # each particles path

    # Need to set the B-filed to grow that the trajectories of SiNPS
    ax = plt.gca()  # Get the current axes

    # Get the axis limits
    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    
    x = np.linspace(x_limits[0], x_limits[1], 10)
    y = np.linspace(y_limits[0], y_limits[1], 10)
    z = np.array([0, z_limits[1], 1])

    # Create a meshgrid for 3D space
    X, Y, Z = np.meshgrid(x, y, z)

    # Magnetic field components [Bx, By, Bz] at each point in the grid
    Bx = direction[0]
    By = direction[1]
    Bz = direction[2]

    vectorScale = np.sqrt((x_limits[0] + x_limits[1])**2 + (y_limits[0] + y_limits[1])**2) # helps scale the B Field quivers

    ax.quiver(X, Y, Z, Bx, By, Bz, length=0.05 * vectorScale, normalize=True, color='b', label="B Field", alpha=0.4) # Bfield direction
    
    # sets the legend's lables to be bright
    legend = ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1)     
    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (μm)')
    ax.set_title('Trajectories of simulated Silicon Nano-Particles')

    return fig, ax


def plotMassDisplacement(position, charge):

    # Seperates the position into different charged
    # This is a slower process than np method but 
    #    is the easiest to get working - had
    #   indexing issues with np methods
    positiveC = np.array([[0, 0, 0]])
    negativeC = np.array([[0, 0, 0]])

    data = np.hstack((position, charge))

    for row in data:
        if row[3] < 0:
            negativeC = np.vstack((row[0:3], negativeC))
        else:
            positiveC = np.vstack((row[0:3], positiveC))

    # Create a histogram of particle distances from the origin
    fig, ax = plt.subplots(figsize=(10,7))

    
    ax.hist(np.linalg.norm(position, axis=1), edgecolor='k', label="All particles", alpha=0.7) # histogram of all particles
    ax.hist(np.linalg.norm(negativeC, axis=1), edgecolor='k', color='b', alpha=0.5, label="Negative Charge") # histrogram of negative chages
    ax.hist(np.linalg.norm(positiveC, axis=1), edgecolor='k', color='r', alpha=0.5, label="Positive Charge") # Histrogram of positive charges

    # Customize the plot (optional)
    ax.set_xlabel('Distance from origin (μm)')
    ax.set_ylabel('Frequency')
    ax.set_title("Simulated displacement of Silicon Nano-particles")
    # ax.set_xlim(0,20)
    ax.legend() 

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