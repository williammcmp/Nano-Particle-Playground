# src/ParticleGenerator.py
from src.Particle import Particle
from tqdm import tqdm
import numpy as np
import random

# Generates there standard test particles
def GenerateTestParticles(Simulation):
    """
    Load a set of predefined test particles into a simulation.

    This function creates a set of predefined Particle objects with specific positions, velocities,
    masses, and charges. These particles are then added to the provided simulation object.

    Parameters:
    - sim (ParticleSimulation): The simulation object to which the test particles will be added.

    Example:
    ```
    sim = ParticleSimulation()
    LoadTestParticles(sim)
    ```

    Output:
    The function creates and adds predefined test particles to the simulation object.

    Returns:
    None
    """
    print("\nLoading Test Particles:")
    # Create a particle and add it to the simulation
    p1 = Particle([0, 0, 0], [0, 1, 3], 1, -1)
    p2 = Particle([0, 0, 0], [0, 1, 3], 1, 1)
    p3 = Particle([0, 0, 0], [0, 1, 3], 1, 0)
    Simulation.AddParticles([p1,p2,p3])

# Creates a large amount of particles
def GenerateParticles(n, Simulation, mode = "Origin",
                      positionX = 0, positionY = 0, 
                      positionZ = 0, massRange = [1, 5],
                      avgEnergy = 3, charged = True):
    """
    Create a specified number of particles and add them to the simulation.

    This function generates random positions (m), velocities (m/s), masses (kg), and charges (-1,0,+1) for a given number of particles
    and then creates Particle objects with these properties. These particles are added to the provided Simulation object.

    Parameters:
    - n (int): The number of particles to create.
    - Simulation (ParticleSimulation): The simulation object to which the particles will be added.

    Example:
    ```
    sim = ParticleSimulation()
    ParticleCreation(n=100, Simulation=sim)
    ```

    Output:
    The function creates and adds `n` particles to the simulation object.

    Returns:
    None
    """



    print(f"\nGenerating {n} Particles:")
    particles = []
    for x in tqdm(range(n), unit=" Particle(s)"):
        if mode == "Origin":
            position = np.array([0,0,0])
        elif mode == "off the wall":
            position = np.array([positionX, positionY, positionZ])
        else:
            x_pos = np.random.normal(positionX)
            y_pos = np.random.normal(positionY)
            z_pos = np.abs(np.random.normal(positionZ))
            position = np.array([x_pos, y_pos, z_pos])
        mass = random.uniform(massRange[0],massRange[1]) # fixed to have smaller masses
        velAvg = np.sqrt(((2/3)*avgEnergy)/mass) # 1/3 needed to allow for energy across all axies
        velocity = np.random.uniform(-velAvg, velAvg, 3)
        velocity[2] = np.abs(velocity[2])

        if charged:
            charge = random.uniform(1.0, 2.0) * mass * random.choice([-1,1]) # implemetns the charge/mass factor for NPs

            particles.append(Particle(position, velocity, mass, charge))
        else: 
            particles.append(Particle(position, velocity, mass))

    Simulation.AddParticles(particles)