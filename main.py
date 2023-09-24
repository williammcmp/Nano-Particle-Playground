# Importing all the required Classes
from Simulation import *
from Particle import *
from Forces import *

# Create a simulation
sim = Simulation()

# Load test particles
# LoadTestParticles(sim)

# Creates the particles
np.random.seed(0)
ParticleCreation(100000,sim)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorenz(np.array([2.0,2.0,0.0])))

# Runns the simulation 
sim.Run(1, 0.01)

# plot the simulation results
sim.Plot()
# sim.PlotPaths()

# TODO add option to run method to re-calcuate the forces at each time step
