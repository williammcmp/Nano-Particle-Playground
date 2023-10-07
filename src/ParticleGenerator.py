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
        else:
            x_pos = np.random.normal(positionX)
            y_pos = np.random.normal(positionY)
            z_pos = np.random.normal(positionZ)
            position = np.array([x_pos, y_pos, z_pos])
        mass = random.uniform(massRange[0],massRange[1]) # fixed to have smaller masses
        velAvg = np.sqrt((2*avgEnergy)/mass)
        velocity = np.random.uniform(-velAvg, velAvg, 3)
        print(velocity)
        if charged:
            charge = random.choice([-1,0, 1])
            particles.append(Particle(position, velocity, mass, charge))
        else: 
            particles.append(Particle(position, velocity, mass))

    Simulation.AddParticles(particles)


# Generates particles in the nano-sized regime
def GenerateNanoParticles(n, Simulation):
    """
    Generates and adds Nano-Particles (modeling SiNPs) to a given simulation.

    This function generates a specified number of Nano-Particles and adds them to a provided Simulation instance.

    Parameters:
    - n (int): The number of Nano-Particles to generate.
    - Simulation (Simulation): The Simulation instance to which the particles will be added.

    Returns:
    None

    Example:
    ```
    sim = Simulation()
    GenerateNanoParticles(100, sim)
    ```
    """
    # 1 meter unit = 10^-9 m
    print(f"\nGenerating {n} Nano-Particles:")
    particles = []
    for x in tqdm(range(n), unit=" Particle(s)"):
        position = np.array([0,0,0]) # all start from origin (nm)
        velocity = np.array([np.random.normal(loc=0, scale=10),np.random.normal(loc=0, scale=10),random.randint(0,10)]) # (nm/s)
        mass = np.abs(np.random.normal(loc=97, scale=25)) # (Kg)
        charge = random.choice([-1,0, 1]) # (c)
        particles.append(Particle(position, velocity, mass, charge))

    Simulation.AddParticles(particles)