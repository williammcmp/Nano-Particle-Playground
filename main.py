# Importing all the required Classes
from Simulation import *
from Particle import *
from Forces import *

# Create a simulation
sim = Simulation()


# Create a particle and add it to the simulation
p1 = Particle([2, 1, 2], [0.3, -0.4, 28], 1, -1)
p2= Particle([1.5, 1.1, 2], [0.3, 0.1, 33], 1, 1)
p3 = Particle([3, 2, 2], [0.2, 0.5, 29], 1, 0)
sim.Particles.append(p1)
sim.Particles.append(p2)
sim.Particles.append(p3)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorenz(np.array([1,0,0])))

# Display information about the system
sim.Display()

#  tick over the simulation
for x in range(2000):
    sim.Update(0.01)

# plot the simulation results
sim.Plot()
