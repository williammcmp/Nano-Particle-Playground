import numpy as np

# force due to gravity
class Gravity:
    """
    Represents the force of gravity in the simulation.

    Attributes:
    - Acceleration (numpy.ndarray): The acceleration due to gravity in [x, y, z] meters per second squared.

    Methods:
    - Apply(particles): Applies the gravitational force to a list of particles.

    """

    def __init__( self, acceleration = 9.8 ):
        """
        Initializes a new Gravity instance.

        Parameters:
        - acceleration (float, optional): The acceleration due to gravity in meters per second squared. Default is 9.8 m/s^2.
        """
        self.name = "Gravity"
        self.Acceleration = np.array([0.0,0.0,-acceleration ])

    # Applies the gravity acceleration into each particle
    def Apply( self, particles):
        """
        Applies the gravitational force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which gravity is applied.
        """

        for particle in particles:
            particle.SumForce = particle.SumForce + (self.Acceleration * particle.Mass)
    
    def Info(self):
        return f"\t{self.name} = {self.Acceleration}\n"

# Viscous Drag Force
class Damping:
    """
    Represents a viscous drag force in the simulation.

    Attributes:
    - Scaling (float): The scaling factor for the drag force.

    Methods:
    - Apply(particles): Applies the viscous drag force to a list of particles.

    """
    def __init__( self, scaling = 1.0 ):
        """
        Initializes a new Damping instance.

        Parameters:
        - scaling (float, optional): The scaling factor for the drag force. Default is 1.0.
        """
        self.Scaling  = scaling

    def Apply( self, particles ):
        """
        Applies the viscous drag force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which the drag force is applied.
        """
        for particle in particles:
            particle.SumForce = particle.SumForce + (particle.Velocity * -self.Scaling)


class Lorenz:
    def __init__( self, bField=np.array([0.0,0.0,0.0]), eField=np.array([0.0,0.0,0.0])):
        self.eField = eField
        self.bField = bField
        self.name = "Lorenz"

    def Apply( self, particles):

        for particle in particles:
            LorenzForce = particle.Charge * self.eField + particle.Charge *(np.cross(particle.Velocity, self.bField))
            particle.SumForce = particle.SumForce + (LorenzForce)

    def Info( self ):
        return f"\tElectric = {self.eField}\n\tMagnetic = {self.bField}\n"
