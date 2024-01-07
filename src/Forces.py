# src/Force.py
import numpy as np
from abc import ABC, abstractclassmethod


class Force(ABC):
    """
    Abstract base class for representing forces in a particle simulation.

    Attributes:
    - Name (str): The name of the force.
    - Magnitude (float): The magnitude of the force.
    - Direction (numpy.ndarray): The direction vector of the force.

    Methods:
    - Apply(particles): Abstract method to apply the force to a list of particles.
    - Field(): Calculates the force field.
    - Info(): Returns a string representation of force information.
    - __str__(): Returns the name of the force.
    """

    def __init__(self, name, magnitude, direction):
        """
        Initializes a new Force instance.

        Parameters:
        - name (str): The name of the force.
        - magnitude (float): The magnitude of the force.
        - direction (numpy.ndarray): The direction vector of the force.
        """
        self.Name = name
        self.Magnitude = magnitude
        self.Direction = direction

    @abstractclassmethod
    def Apply(self, particles):
        """
        Abstract method to apply the force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which the force is applied.
        """
        pass

    def Field(self):
        """
        Calculates the force field.

        Returns:
        - numpy.ndarray: The force field vector.
        """
        return self.Magnitude * self.Direction

    def Info(self):
        """
        Returns a string representation of force information.

        Returns:
        - str: String representation of force information.
        """
        return f'{self.Name} = {self.Field}'

    def __str__(self):
        """
        Returns the name of the force.

        Returns:
        - str: The name of the force.
        """
        return self.Name


# force due to gravity
class Gravity(Force):
    """
    Represents the force of gravity in the simulation.

    Attributes:
    - Acceleration (numpy.ndarray): The acceleration due to gravity in [x, y, z] meters per second squared.

    Methods:
    - Apply(particles): Applies the gravitational force to a list of particles.
    """

    def __init__(self, magnitude=9.8):
        """
        Initializes a new Gravity instance.

        Parameters:
        - magnitude (float, optional): The acceleration due to gravity in meters per second squared. Default is 9.8 m/s^2.
        """
        name = "Gravity"
        direction = np.array([0.0, 0.0, -1])
        super().__init__(name, magnitude, direction)

    def Apply(self, particles):
        """
        Applies the gravitational force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which gravity is applied.
        """
        for particle in particles:
            particle.SumForce = particle.SumForce + (self.Field() * particle.Mass)
    
    # def NanoApply( self, particles):
    #     """
    #     Applies the gravitational force to a list of nano-particles.

    #     Parameters:
    #     - particles (list): A list of Particle objects to which gravity is applied.
    #     """
    #     for particle in particles:
    #         particle.SumForce = particle.SumForce + (self.Acceleration * particle.Mass * 1e9)
    

# Viscous Drag Force
class Damping(Force):
    """
    Represents the viscous drag force in the simulation.

    Methods:
    - Apply(particles): Applies the viscous drag force to a list of particles.
    """

    def __init__(self, magnitude=1.0, direction=np.array([0.0, 0.0, -1])):
        """
        Initializes a new Damping instance.

        Parameters:
        - magnitude (float, optional): The scaling factor for the drag force. Default is 1.0.
        """
        name = "Damping"
        super().__init__(name, magnitude, direction)

    def Apply(self, particles):
        """
        Applies the viscous drag force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which the drag force is applied.
        """
        for particle in particles:
            particle.SumForce = particle.SumForce + (particle.Velocity * -self.Field())



class Magnetic(Force):
    """
    Represents the magnetic force in the simulation.

    Methods:
    - Apply(particles): Applies the magnetic force to a list of particles.
    """

    def __init__(self, magnitude=1.0, direction=np.array([1, 0.0, 0.0])):
        name = "Magnetic"
        super().__init__(name, magnitude, direction)

    def Apply(self, particles):
        for particle in particles:
            particle.SumForce = particle.SumForce + (particle.Charge * (np.cross(particle.Velocity, self.Field())))


class Electric(Force):
    """
    Represents the electric force in the simulation.

    Methods:
    - Apply(particles): Applies the electric force to a list of particles.
    """

    def __init__(self, magnitude=1.0, direction=np.array([1, 0.0, 0.0])):
        name = "Electric"
        super().__init__(name, magnitude, direction)

    def Apply(self, particles):
        for particle in particles:
            particle.SumForce = particle.SumForce + (particle.Charge * self.Field())
    
class Barrier(Force):
    """
    Barrier represents a wall or ground plane in the simulation that will reflect the particles upon contact.

    @ivar Plane: The normal vector of the barrier plane.
    @type Plane: numpy.ndarray
    @ivar Offset: The offset from the origin that the barrier plane exists.
    @type Offset: numpy.ndarray

    @param damping: The damping factor for the barrier.
    @type damping: float
    @param plane: The normal vector of the barrier plane.
    @type plane: numpy.ndarray
    @param offset: The offset from the origin that the barrier plane exists.
    @type offset: numpy.ndarray
    """

    def __init__(self, damping=1.0, plane=np.array([0.0, 0.0, 1.0]), offset=np.array([0.0, 0.0, 0.0])):
        """
        Initialize the Barrier object.

        @param damping: The damping factor for the barrier.
        @type damping: float
        @param plane: The normal vector of the barrier plane.
        @type plane: numpy.ndarray
        @param offset: The offset from the origin that the barrier plane exists.
        @type offset: numpy.ndarray
        """
        name = "Barrier"
        super().__init__(name, damping, plane)
        self.Plane = plane
        self.Offset = offset

    def Apply(self, particles):
        """
        Applies the barrier to particles by reversing their position and velocity if they penetrate the directional plane.

        @param particles: List of particles to apply the barrier to.
        @type particles: list
        """
        for particle in particles:
            # d = normal . (particle - offset)
            distance = np.dot(self.Plane, particle.position - self.Offset)  # gets the distance of the particle from the normal of the plane
            if distance <= 0:
                # TODO: check that the particles don't get stuck behind the barrier -> the velocities keep flipping as we are negative to the plane's normal
                particle.Velocity = particle.Velocity * (self.Field() * -1)  # self.Field gives the bounce off the plane, already at the barrier to bounce off
