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
ParticleCreation(1000,sim)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorenz(np.array([0,0,10.0])))

# Runns the simulation 
sim.Run(5, 0.01, False)

# plot the simulation results
sim.Plot()
# sim.PlotPaths()
