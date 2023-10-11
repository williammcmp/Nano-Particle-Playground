# Importing all the required Classes
from src.Simulation import *
from src.Particle import *
from src.Forces import *
from src.ParticleGenerator import *

# Create a simulation
sim = Simulation()

# Load test particles
# GenerateTestParticles(sim)

# Creates the particles
np.random.seed(0)
GenerateParticles(10000,sim)

# Add gravity force to the simulation
sim.AddForce([Gravity()])
# sim.AddForce([Lorentz(np.array([0,0,10.0]))])

# Runns the simulation 
sim.FastRun(5, 0.01)

# plot the simulation results
sim.Plot()
# sim.PlotPaths()
