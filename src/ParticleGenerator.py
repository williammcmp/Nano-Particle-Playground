# src/ParticleGenerator.py
from src.Particle import Particle
from tqdm import tqdm
import numpy as np
import scipy as sp
import random
import json

from src.DataLoader import *

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
    # p1 = Particle([0, 0, 0], [0, 1, 3], 9, -10)
    # p2 = Particle([0, 0, 0], [0, 1, 3], 9, 10)
    # p3 = Particle([0, 0, 0], [0, 1, 3], 9, 0)
    p1 = Particle([0, 0, 0.01], [1, 0, 3], 1, -1)
    p2 = Particle([0, 0, 0.01], [1, 0, 3], 1, 1)
    p3 = Particle([0, 0, 0.01], [1, 0, 3], 1, 0)
    Simulation.AddParticles([p1,p2,p3])

# Creates a large amount of particles
def GenerateParticles(n, Simulation, mode = "Origin",
                      positionX = 0, positionY = 0, 
                      positionZ = 0, massRange = [1, 5],
                      avgEnergy = 3, charged = True,  chargedNev = True, chargedPos = True):
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

    if mode == "load":
        particles = LoadParticleSettings()
    else:
        print(f"\nGenerating {n} Particles:")
        particles = []
        for x in tqdm(range(n), unit=" Particle(s)"):
            if mode == "Origin":
                position = np.array([0,0,0.01]) # a small inital offest from the ground plane -> avoids intial bouncing 
            elif mode == "off the wall":
                position = np.array([positionX, positionY, positionZ])
            else:
                x_pos = np.random.normal(positionX)
                y_pos = np.random.normal(positionY)
                position = np.array([x_pos, y_pos, 0.001]) # all particles will start at z = 0

            mass = random.uniform(massRange[0],massRange[1]) # fixed to have smaller masses
            velAvg = np.sqrt(((2/3)*avgEnergy)/mass) # 1/3 needed to allow for energy across all axies
            velocity = np.random.uniform(-velAvg, velAvg, 3)
            velocity[2] = np.abs(velocity[2])


            

            if charged:
                charge = random.uniform(1.0, 2.0) * mass * random.choice([-1 * chargedNev,1 * chargedPos]) # implemetns the charge/mass factor for NPs
                # charge = random.uniform(1.0, 2.0) * mass # implemetns the charge/mass factor for NPs
                

                particles.append(Particle(position, velocity, mass, charge))
            else: 
                particles.append(Particle(position, velocity, mass))

    Simulation.AddParticles(particles)





def pGen (n, size, energy, reduceZ, randomness):
    mass = [ParticleShericalMass(size[0]), ParticleShericalMass(size[1])]
    p_mass = np.random.uniform(mass[0] , mass[1], n)
    p_positions = np.random.randn(n,2) * 1e-6 # scale down the positions to be at the micron scale

    p_velocity = calVelocity(p_mass, p_positions * 1e-3, energy, reduceZ, randomness)
    
    zeros = np.zeros((p_positions.shape[0], 1)) + 0.0000000001
    p_positions = np.hstack((p_positions, zeros))
    
    dic = {
        "pos" : p_positions,
        "mass" : p_mass, 
        "vel": p_velocity
    }
    return dic 

# Looads the from the pGen
def pLoad(settings):

    generatedSettings = pGen(settings['particleNumber'], settings['particleSize'], settings['particleEnergy'], settings['useNonConstantZ'], settings['randomness'])

    position = generatedSettings['pos'] 
    mass = generatedSettings['mass']
    velocity = generatedSettings['vel']

    particleCount = settings['particleNumber']
    particles = []
    for row in range(particleCount):
        charge = random.uniform(1.0, 2.0) * random.choice([-3,3])
        particles.append(Particle(position[row], velocity[row], mass[row], charge))

    return particles
    

def calVelocity(mass, position, energy, reduceZ = False, randomness=False):

    velMag = np.sqrt(2 * (energy * 100) / mass) # 1/3 needed to allow for energy across all axies
    
    
    if reduceZ:
        z = np.sqrt(9 - (position[:,0]**2 + position[:,1]**2)) # Reduces the velocity in the Z mag when further away from the origin (pre normalised z is always 1)
    else:
        z = np.ones((position.shape[0], 1)) # The virtical compoent of the vecotr before normalisation is always 1

    zMag = z.reshape(-1, 1) # Allows z postional values to be hstacked on the position array

    velDir = np.hstack((position * 1e5, zMag)) # 1e4 is to scale up the position from the origin
    velNorm = velDir / np.linalg.norm(velDir, axis=1, keepdims=True) # ensure normalization along the correct axis
    Velocity = velMag.reshape(-1, 1) * velNorm

    if randomness:
        a = np.random.randn(position.shape[0],3) # Random offset in the particles inital velocity
        Velocity = Velocity + a * 1e-3 # apply the random offset to the particles inital velocity. 1e-3 is to scale down the random offset -> designed to run at the nm scales
        Velocity[:, 2] = np.abs(Velocity[:, 2]) # makes Vz positive

    return Velocity
        

def LoadParticleSettings():
    loaded_data = load_from_json()

    if loaded_data is not None:
        print("Loaded data:")
        # print(loaded_data)
        particles = pLoad(loaded_data)
    
    return particles

# Calcuates the mass of a spherical particle, default uses Silicon density. Retuens in standard Units
def ParticleShericalMass(diamater, density = 2330): 
    """
    Calculate the mass of a spherical particle.

    This function computes the mass of a spherical particle based on its diameter and density, where the default density is set to that of Silicon.

    Parameters:
    - diameter (float): The diameter of the spherical particle.
    - density (float, optional): The density of the material composing the particle. Default is 2330 kg/m^3, the density of Silicon.

    Returns:
    - float: The mass of the spherical particle in standard units (kilograms).

    Example:
    ```
    particle_mass = ParticleSphericalMass(0.1)  # Calculates the mass of a particle with a diameter of 0.1 meters
    ```
    """
    mass = (3/4) * np.pi * ((diamater / 2) ** 2) * density

    return mass


    
    
