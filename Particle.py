import numpy as np

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

    def __init__(self, position, velocity, mass, forceSum=np.array([0, 0, 0])):
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

    def Display(self):
        """
        Returns the current position of the particle.

        Returns:
        - numpy.ndarray: The current position of the particle in [x, y, z] meters.
        """
        return self.Position

    
