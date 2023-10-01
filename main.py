# Importing all the required Classes
from src.Simulation import *
from src.Particle import *
from src.Forces import *
from src.ParticleGenerator import *

# Create a simulation
sim = Simulation()

# Load test particles
# LoadTestParticles(sim)

# Creates the particles
np.random.seed(0)
GenerateParticles(1000,sim)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorentz(np.array([0,0,10.0])))

# Runns the simulation 
sim.Run(5, 0.01, False)

# plot the simulation results
sim.Plot()
# sim.PlotPaths()
