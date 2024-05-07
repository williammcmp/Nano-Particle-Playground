# src/Force.py
import numpy as np
from abc import ABC, abstractclassmethod

from numpy.core.multiarray import array as array


class Force(ABC):
    """
    Abstract base class for representing forces in a particle simulation.

    Attributes:
    - Name (str): The name of the force.
    - Magnitude (float): The magnitude of the force.
    - Direction (numpy.ndarray): The normalised direction vector of the force.

    Methods:
    - Apply(particles): Abstract method to apply the force to a list of particles.
    - Field(): Calculates the force field.
    - UpdateField(field): Updates the normalised direction vector and magitude of the force.
    - Info(): Returns a string representation of force information.
    - __str__(): Returns the name of the force.
    """

    def __init__(self, name, magnitude, direction, units = "N"):
        """
        Initializes a new Force instance.

        Parameters:
        - name (str): The name of the force.
        - magnitude (float): The magnitude of the force.
        - direction (numpy.ndarray): The normalised direction vector of the force. 
        """
        self.Name = name
        self.Magnitude = magnitude
        self.Direction = direction / np.linalg.norm(direction)
        self.Units = units

    @abstractclassmethod
    def Apply(self, particles):
        """
        Abstract method to apply the force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which the force is applied.
        """
        pass


    def Diagram( self, ax ):
        """
        The the force to the free body diagram.
        """
        pass


    def Field(self):
        """
        Calculates the force field.

        Returns:
        - numpy.ndarray: The force field vector.
        """
        return self.Magnitude * self.Direction

    def UpdateField (self, field):
        """
        Updates the force field by saving its magnitude and normalized direction.

        Parameters:
        - field (numpy.ndarray): The force field to be updated.

        Returns:
        None
        """
        self.Magnitude = np.linalg.norm(field)

        # cant normalise a zero vector
        if np.all(field) != 0:
            self.Direction = field / np.linalg.norm(field) # the direction must be normalsed 
        else:
            self.Direction = field


    def Info(self):
        """
        Returns a string representation of force information.

        Returns:
        - str: String representation of force information.
        """
        return f'{self.Name} = {self.Field()} ({self.Units})'

    def __str__(self):
        """
        Returns the name of the force.

        Returns:
        - str: The name of the force.
        """
        return self.Name
    
    def Dict( self ):
        dict = {f'{self.Name}': [self.Direction, self.Magnitude, self.Field(), self.Units]}
        return dict


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
        units = "m/s^2"
        super().__init__(name, magnitude, direction, units)

    def Apply(self, particles):
        """
        Applies the gravitational force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which gravity is applied.
        """
        for particle in particles:
            particle.SumForce = particle.SumForce + (self.Field() * particle.Mass)

    def Diagram( self, ax):
        ax.quiver(0, 0, 4,
                  0, 0, -4, 
                  length = 1, label = self.Name, color = 'red', alpha = 1)
    
    # def NanoApply( self, particles):
    #     """
    #     Applies the gravitational force to a list of nano-particles.

    #     Parameters:
    #     - particles (list): A list of Particle objects to which gravity is applied.
    #     """
    #     for particle in particles:
    #         particle.SumForce = particle.SumForce + (self.Acceleration * particle.Mass * 1e9)
    

# Viscous Drag Force
# TODO: make this drag force more physically realistic 
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
            particle.SumForce = particle.SumForce - self.Field()



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

    def Diagram( self, ax):
        x, y, z = np.meshgrid(np.linspace(-4, 4, 3), 
                      np.linspace(-4, 4, 3),
                      np.linspace(-0, 4, 3))
        
        u, v, w = self.Field() * 50 # added scalar to make quivers larger for ploting

        ax.quiver(x, y, z, u, v, w,
                  label = self.Name, color = 'blue', alpha = 0.7)



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
    
    def Diagram( self, ax):
        x, y, z =  np.meshgrid(np.linspace(-4, 4, 3), 
                      np.linspace(-4, 4, 3),
                      np.linspace(-0, 4, 3))
        
        u, v, w = self.Field() * 2

        ax.quiver(x, y, z, u, v, w,
                  label = self.Name, color = 'green', alpha = 0.7)
    
class Barrier(Force):
    """
    Barrier represents a wall or ground plane in the simulation that will reflect the particles upon contact. By default, the Barrier object is setup as a ground plane

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

    def __init__(self, damping=0, plane=np.array([0.0, 0.0, 1.0]), offset=np.array([0.0, 0.0, 0.0]), units="m"):
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
        super().__init__(name, damping, plane, units)
        self.Plane = plane
        self.Offset = offset

    def Apply(self, particles):
        """
        Applies the barrier to particles by reversing velocity if they penetrate the directional plane. Their postion is also set to be on the correct side of the barrier

        @param particles: List of particles to apply the barrier to.
        @type particles: list
        """
        for particle in particles:
            # d = normal . (particle - offset)
            distance = np.dot(self.Plane, particle.Position - self.Offset)  # gets the distance of the particle from the normal of the plane

            if distance < 0:
                # TODO: the offset logic is not 100% working and can result in unexpected results

                n = self.Plane / np.linalg.norm(self.Plane) #normalises the reflection plane normal
                particle.Position = particle.Position - (n * distance)

                reflection = particle.Velocity - 2 * (np.dot (particle.Velocity, n) * n)  # the reflected velocity vector off the planes normal
                particle.Velocity = reflection * self.Magnitude # Magitude of the bounceness of the plane
    
class GroundPlane(Force):
    def __init__(self, damping=0.0):
        name = 'GroundPlane'
        super().__init__(name, damping, np.array([0, 0, 1]), "m")

    def Apply(self, particles):

        for particle in particles:

            if( particle.Position[2] < 0 ):
                particle.Position[2] = 0 # reset the particle's position to above the ground plane

                particle.Velocity = particle.Velocity * np.array([self.Magnitude, self.Magnitude, -1*self.Magnitude])

    def Diagram(self, ax):
        # Create a ground plane
        X, Y = np.meshgrid(np.linspace(-5, 5, 50), np.linspace(-5, 5, 50))
        Z = np.zeros(X.shape)  # Ground plane at z=0

        # Plot the ground plane
        ax.plot_surface(X, Y, Z, color='tan', alpha=0.5)