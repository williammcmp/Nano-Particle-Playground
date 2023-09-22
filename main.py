# Importing all the required Classes
from Simulation import *
from Particle import *
from Forces import *
import time

# Create a simulation
sim = Simulation()


# Create a particle and add it to the simulation
# p1 = Particle([2, 1, 2], [0.3, -0.4, 28], 1, -1)
# p2= Particle([1.5, 1.1, 2], [0.3, 0.1, 33], 1, 1)
# p3 = Particle([3, 2, 2], [0.2, 0.5, 29], 1, 0)
# sim.Particles.append(p1)
# sim.Particles.append(p2)
# sim.Particles.append(p3)

ParticleCreation(400,sim)

# Add gravity force to the simulation
sim.Forces.append(Gravity())
sim.Forces.append(Lorenz(np.array([1,2,0.5])))

# Display information about the system
# sim.Display()

#  tick over the simulation
startTime = time.time()
for x in range(2000):
    sim.Update(0.01)

runTime = time.time() - startTime
print(f"single threading: {runTime}s")

sim.Particles = []
print(len(sim.Particles))

ParticleCreation(400,sim)

# sim.Display()

#  tick over the simulation
startTime = time.time()
for x in range(2000):
    sim.UpdateMultiThread(0.01)

runTime = time.time() - startTime
print(f"Multi threading: {runTime}s")


# plot the simulation results
sim.Plot()
