import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import pandas as pd
# import seaborn as sns

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
        "Magnetic Field Across the Page -Y": "BFieldAcrossDown",
        "Magnetic Field Across the Page +Y": "BFieldAcrossUp"
    }

    dataSeries = ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page", "Magnetic Field Across the Page -Y", "Magnetic Field Across the Page +Y"]

    for series in dataSeries:
        data = load_experimental_data(dataType[series])

        x_mean = data["X"].mean()
        y_mean = data["Y"].mean()

        if x_mean < 0:
            x_mean -= 3000
        else:
            x_mean += 3000


        # Data is with respect to edge of the crator
        if y_mean < 0:
            y_mean -= 3000
        else:
            y_mean += 3000

        ax.scatter(x_mean * 1e-3, y_mean * 1e-3, label=f"{series}", marker='1', s = 350)
    
    ax.legend()

    return fig, ax

def plotExperimentalDistributions(fig, ax):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn", 
        "Magnetic Field Across the Page -Y": "BFieldAcrossDown",
        "Magnetic Field Across the Page +Y": "BFieldAcrossUp"
    }

    dataSeries = ["No Magentic Field", "Magnetic Field out of the Page", "Magnetic Field into the Page", "Magnetic Field Across the Page -Y", "Magnetic Field Across the Page +Y"]

    bin_width = 10  # Adjust the bin width as needed
    bins = np.arange(0, 100, bin_width)
    for series in dataSeries:

        data = load_experimental_data(dataType[series])
        
        ax.hist(data["Width"], bins=bins, edgecolor='k', alpha=0.7, label=series)


    ax.legend()

    return fig, ax

def plotExperimentalDistribution(dataSeries, fig, ax):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn", 
        "Magnetic Field Across the Page -Y": "BFieldAcrossDown",
        "Magnetic Field Across the Page +Y": "BFieldAcrossUp"
    }

    bin_width = 10  # Adjust the bin width as needed
    bins = np.arange(0, 100, bin_width)


    data = load_experimental_data(dataType[dataSeries])
    
    ax.hist(data["Width"], bins=bins, edgecolor='k', alpha=0.7, label=dataSeries)

    ax.legend()

    return fig, ax

    
        

def plotExperimentalData(dataSeries):
    # This dictionary makes it easer to load the data files
    dataType = {
        "No Magentic Field" : "NoBField",
        "Magnetic Field out of the Page": "BFieldOut", 
        "Magnetic Field into the Page": "BFieldIn", 
        "Magnetic Field Across the Page -Y": "BFieldAcrossDown",
        "Magnetic Field Across the Page +Y": "BFieldAcrossUp"
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

    circle = plt.Circle((0, 0), 1e-6, color='r', fill=False)
    ax.add_artist(circle)

    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Simulated position of Silicion Nano-Particles')
    ax.grid(True)

    ax.set_xlim(-1e-5, 1e-5)
    ax.set_ylim(-1e-5, 1e-5)

    return fig, ax

def plotRadialPosition(position):
    radius = np.sqrt(position[:,0] ** 2 + position[:,1] ** 2) * 1e6
    radius_filted = radius[radius > 2]

    fig, ax = plt.subplots(figsize=(10,6))

    radius_data = {'All Particles': radius,
                   'Outside ablation site': radius_filted}

    # sns.displot(radius_data, kind='kde', bw_adjust=0.5)

    # sns.displot(radius, kind='kde', bw_adjust=0.5, label="all particles")
    # sns.displot(radius_filted, kind='kde', bw_adjust=0.5, label="outside ablation site")
    plt.axvline(x=2, color='r', linestyle='--')

    # # Shade the area to the left of the vertical line
    plt.axvspan(-5, 2, alpha=0.2, color='red')

    # # Add text to the shaded area
    plt.text(2.5 , 0.10, 'Ablation Site', color='red', fontsize=12)

    plt.xlim(-0.5, 25)
    plt.ylim(bottom=-0.001)
    plt.ylabel("Density (particle/µm^2)")
    plt.xlabel("Radial position (µm)")
    plt.legend()
    plt.grid()

    st.pyplot()


def plotSimulatedMassHistogram(mass):
    fig, ax = plt.subplots(figsize=(10,7))
    ax.hist(np.linalg.norm(mass*10, axis=1), bins=10, edgecolor='k', alpha=0.7)

    ax.set_xlabel('Particle diamater (nm)')
    ax.set_ylabel('Frequency')
    ax.set_title("Simulated Silicon nano-particle size")

    return fig, ax


def plotTrajectories(simulation):

   
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
    
    x = np.linspace(x_limits[0], x_limits[1], 5)
    y = np.linspace(y_limits[0], y_limits[1], 5)
    z = np.array([0, z_limits[1], 1])

    # Create a meshgrid for 3D space
    X, Y, Z = np.meshgrid(x, y, z)

    # Magnetic field components [Bx, By, Bz] at each point in the grid
    Bx = 0
    By = 0
    Bz = 0

    # Update the direction if the magnetic force is used
    # if simulation.HasForce("Magnetic"):
    #     direction = simulation.GetForce("Magnetic")[0].Field()
    #     print(direction)
    #     Bx = direction[0]
    #     By = direction[1]
    #     Bz = direction[2]

    # vectorScale = np.sqrt((x_limits[0] + x_limits[1])**2 + (y_limits[0] + y_limits[1])**2) # helps scale the B Field quivers

    # ax.quiver(X, Y, Z, Bx, By, Bz, length=0.1 * vectorScale, normalize=True, color='b', label="B Field", alpha=0.4) # Bfield direction
    
    # sets the legend's lables to be bright
    legend = ax.legend()
    for lh in legend.legendHandles:
        lh.set_alpha(1)     
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Trajectories of simulated Silicon Nano-Particles')
    # ax.set_xlim([-1e-5, 1e-5])
    # ax.set_ylim([-1e-5, 1e-5])
    # ax.set_zlim([0,0.005])

    radius = 2 * 1e-6
    center = (0, 0, 0)
    num_points = 100

    # Parametric equations for a circle in 3D space
    theta = np.linspace(0, 2*np.pi, num_points)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    z = center[2] + np.zeros_like(theta)  # All z-coordinates are zeros (lies in XY plane)

    ax.plot(x, y, z, color='r')

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

    bin_width = 3  # Adjust the bin width as needed
    bins = np.arange(0, 40, bin_width)

    ax.hist(np.linalg.norm(position, axis=1), edgecolor='k', label="All particles", bins=bins, alpha=0.7) # histogram of all particles
    ax.hist(np.linalg.norm(negativeC, axis=1), edgecolor='k', color='b', alpha=0.5, bins=bins, label="Negative Charge") # histrogram of negative chages
    ax.hist(np.linalg.norm(positiveC, axis=1), edgecolor='k', color='r', alpha=0.5, bins=bins, label="Positive Charge") # Histrogram of positive charges

    # Customize the plot (optional)
    ax.set_xlabel('Distance from origin (μm)')
    ax.set_ylabel('Frequency')
    ax.set_title("Simulated displacement of Silicon Nano-particles")
    # ax.set_xlim(0,20)
    ax.legend() 

    return fig, ax
    

def plotForceTrajectories(simulation, direction):
    # Create a histogram of particle distances from the origin
    fig, ax = plt.subplots(figsize=(10,7))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, projection='3d')


    x_limits = ax.get_xlim()
    y_limits = ax.get_ylim()
    z_limits = ax.get_zlim()
    
    x = np.linspace(x_limits[0], x_limits[1], 5)
    y = np.linspace(y_limits[0], y_limits[1], 5)
    z = np.array([0, z_limits[1], 1])

    # Create a meshgrid for 3D space
    X, Y, Z = np.meshgrid(x, y, z)

    # Magnetic field components [Bx, By, Bz] at each point in the grid
    Bx = direction[0]
    By = direction[1]
    Bz = direction[2]

    vectorScale = np.sqrt((x_limits[0] + x_limits[1])**2 + (y_limits[0] + y_limits[1])**2) # helps scale the B Field quivers

    # ax.quiver(X, Y, Z, Bx, By, Bz, length=0.01 * vectorScale, normalize=True, color='b', label="B Field", alpha=0.4) # Bfield direction
    colors = {1: "r", 2: "g", 3:"b"}
    count = 1
    for particle in simulation.Particles:
            
        ax.plot(particle.History[:,0], particle.History[:,1], particle.History[:,2], c=colors[count], linestyle="--") # plots the path for each particle
        for i in np.arange(0,len(particle.History), 40):
            ax.scatter(particle.History[i,0], particle.History[i,1], particle.History[i,2], c=colors[count], s=10) # key potints for the particles trajectory
            
            x_pos = particle.History[i,0]
            y_pos = particle.History[i,1]
            z_pos = particle.History[i,2]
            
            F_mag = particle.Charge * (np.cross(particle.HistoryVel[i,:], direction))
            print(F_mag)

            # B field force vector
            ax.quiver(x_pos, y_pos, z_pos, F_mag[0], F_mag[1], F_mag[2], length=np.linalg.norm(F_mag)/500, normalize=True)

            # Gravity
            ax.quiver(x_pos, y_pos, z_pos, 0, 0, -9.8, length=np.linalg.norm(np.array([0,0,-9.8]))/500, normalize=True, color='g')
        
        count += 1
        if count > 3:
            break

    ax.set_xlabel('X (μm)')
    ax.set_ylabel('Y (μm)')
    ax.set_zlabel('Z (μm)')

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

# TODO refactor to be more maintainable
def PlotBeamFocal(ax, beam_width, z_air, z_silicon, z_abs_depth, z_MPI_depth):
    
    x = np.linspace(-beam_width, beam_width, 200)  # Limit x to the range where the square root is defined
    
    # Calacuate the Air focal 
    eps_air = np.sqrt(z_air ** 2 * (1 - (x ** 2) / beam_width ** 2))

    # Plot Air focal spot
    ax.plot(x, eps_air, color = 'blue', label = "Air")

    # Calculate the Silicon focus plot
    eps_silicon = -np.sqrt(z_silicon ** 2 * (1 - (x ** 2) / beam_width ** 2))
    eps_abs_depth = -np.sqrt(z_abs_depth ** 2 * (1 - (x ** 2) / beam_width ** 2))
    eps_MPI_depth = -np.sqrt(z_MPI_depth ** 2 * (1 - (x ** 2) / (beam_width * 0.5) ** 2))

    # Plot Silicon focual spot
    ax.plot(x, eps_silicon, color = 'green', label = "Silicon")
    ax.plot(x, eps_abs_depth, color = 'orange', label = "Thermal depth")
    ax.plot(x, eps_MPI_depth, color =  'purple', label = "MPI depth")


    # Calcuate beam profiles
    z = np.linspace(0, (z_air + z_air) / 2, 200)
    beam_air = beam_width * np.sqrt(1 + (z ** 2 / z_air ** 2))
    beam_silicon = beam_width * np.sqrt(1 + (z ** 2 / z_silicon ** 2))

    ax.plot(beam_air, z, color = "lightgray", linestyle='dotted', label = 'Beam Width')
    ax.plot(-beam_air, z, color = "lightgray", linestyle='dotted')
    ax.plot(beam_silicon, -z, color = "lightgray", linestyle='dotted')
    ax.plot(-beam_silicon, -z, color = "lightgray", linestyle='dotted')

    return ax