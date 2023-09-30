from tqdm import tqdm
import numpy as np
import random

class Particle:
    """
    Represents an individual particle in the simulation.

    Attributes:
    - Position (numpy.ndarray): The position of the particle in [x, y, z] meters.
    - Velocity (numpy.ndarray): The velocity of the particle in [x, y, z] meters per second.
    - SumForce (numpy.ndarray): The sum of forces acting on the particle in [x, y, z] newtons.
    - Mass (float): The mass of the particle in kilograms.

    Methods:
    - Display(): Returns the current position of the particle.
    """

    def __init__(self, position, velocity, mass, charge=0, forceSum=np.array([0, 0, 0])):
        """
        Initializes a new Particle instance.

        Parameters:
        - position (list or numpy.ndarray): The initial position of the particle in [x, y, z] meters.
        - velocity (list or numpy.ndarray): The initial velocity of the particle in [x, y, z] meters per second.
        - mass (float): The mass of the particle in kilograms.
        - forceSum (numpy.ndarray, optional): The initial sum of forces acting on the particle in [x, y, z] newtons. Default is [0, 0, 0].
        """

        self.Position = np.array(position)
        self.Velocity = np.array(velocity)
        self.SumForce = np.array(forceSum)
        self.Mass = mass
        self.Charge = charge
        self.History = position

    def Save(self):
        """
        Saves the positional history of the particle. Used for ploting the particles' path
        """
        self.History = np.vstack((self.History, self.Position))

    def Display(self):
        """
        Returns the current position of the particle.

        Returns:
        - numpy.ndarray: The current position of the particle in [x, y, z] meters.
        """
        return self.Position
    

# Creates a large amount of particles
def GenerateParticles(n, Simulation):
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
    for x in tqdm(range(n), unit=" Particle(s)"):
        position = np.random.uniform(0, 3, 3) + np.array([0,0,1])
        velocity = np.random.uniform(-2, 2, 3)
        mass = random.randrange(1,5) # fixed to have smaller masses
        charge = random.choice([-1,0, 1])
        Simulation.Particles.append(Particle(position, velocity, mass, charge))


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
    Simulation.Particles.append(p1)
    Simulation.Particles.append(p2)
    Simulation.Particles.append(p3)

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
    print(f"\nGenerating {n} Nano-Particles:")
    for x in tqdm(range(n), unit=" Particle(s)"):
        # between 20 mu meters
        position = np.array([0.0000001,0,0]) # all start from origin
        velocity = np.array([np.random.normal(loc=0, scale=10),np.random.normal(loc=0, scale=10),random.randint(0,10)]) * 1e-7 
        mass = np.abs(np.random.normal(loc=100, scale=25))* 1e-19  # mass of particles (~ 10^-19kg)
        charge = 0;
        Simulation.Particles.append(Particle(position, velocity, mass, charge))

