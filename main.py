# Importing all the required Classes
from Simulation import *
from Particle import *
from Forces import *

# Create a simulation
sim = Simulation()

# Load test particles
# LoadTestParticles(sim)

# Creates the particles
ParticleCreation(100,sim)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorenz(np.array([1.0,0.0,0.0])))

# Runns the simulation 
sim.Run(2, 0.01)

# plot the simulation results
sim.Plot()
