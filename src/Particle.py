# src/Particle.py
import numpy as np

class Particle:
    """
    Represents an individual particle in the simulation.

    Attributes:
    - Position (numpy.ndarray): The position of the particle in [x, y, z] meters.
    - Velocity (numpy.ndarray): The velocity of the particle in [x, y, z] meters per second.
    - SumForce (numpy.ndarray): The sum of forces acting on the particle in [x, y, z] newtons.
    - Mass (float): The mass of the particle in kilograms.
    - Charge (float): The charge of the particle. Default is 0.
    - History (numpy.ndarray): The positional history of the particle.
    - HistoryVel (numpy.ndarray): The velocity history of the particle.

    Methods:
    - __init__(position, velocity, mass, charge=0, forceSum=np.array([0, 0, 0])): Initializes a new Particle instance.
    - Save(): Saves the positional history of the particle. Used for plotting the particles' path.
    - Display(): Returns the current position of the particle.
    - __str__(): Returns a string representation of the particle, including its position, velocity, sum of forces, mass, and charge.

    @param position: The initial position of the particle in [x, y, z] meters.
    @type position: list or numpy.ndarray
    @param velocity: The initial velocity of the particle in [x, y, z] meters per second.
    @type velocity: list or numpy.ndarray
    @param mass: The mass of the particle in kilograms.
    @type mass: float
    @param charge: The charge of the particle. Default is 0.
    @type charge: float
    @param forceSum: The initial sum of forces acting on the particle in [x, y, z] newtons. Default is [0, 0, 0].
    @type forceSum: numpy.ndarray, optional
    """

    def __init__(self, position, velocity, mass, charge=0, forceSum=np.array([0, 0, 0])):
        """
        Initializes a new Particle instance.

        @param position: The initial position of the particle in [x, y, z] meters.
        @type position: list or numpy.ndarray
        @param velocity: The initial velocity of the particle in [x, y, z] meters per second.
        @type velocity: list or numpy.ndarray
        @param mass: The mass of the particle in kilograms.
        @type mass: float
        @param charge: The charge of the particle. Default is 0.
        @type charge: float
        @param forceSum: The initial sum of forces acting on the particle in [x, y, z] newtons. Default is [0, 0, 0].
        @type forceSum: numpy.ndarray, optional
        """

        self.Position = np.array(position)
        self.Velocity = np.array(velocity)
        self.SumForce = np.array(forceSum)
        self.Mass = mass
        self.Charge = charge
        self.History = position
        self.HistoryVel = velocity

    def Save(self):
        """
        Saves the positional history of the particle. Used for plotting the particles' path.
        """

        self.History = np.vstack((self.History, self.Position))
        # self.HistoryVel = np.vstack((self.HistoryVel, self.Velocity))

    def Display(self):
        """
        Returns the current position of the particle.

        @return: The current position of the particle in [x, y, z] meters.
        @rtype: numpy.ndarray
        """
        return self.Position

    def __str__(self):
        """
        Returns a string representation of the particle, including its position, velocity, sum of forces, mass, and charge.

        @return: String representation of the particle.
        @rtype: str
        """
        return f"Position : {self.Position}\nVelocity: {self.Velocity}\nSum Force: {self.SumForce}\nMass: {self.Mass}\nCharge: {self.Charge}"
