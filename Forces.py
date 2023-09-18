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

        self.Acceleration = np.array([ 0.0, 0.0, -acceleration ])

    # Applies the gravity acceleration into each particle
    def Apply( self, particles):
        """
        Applies the gravitational force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which gravity is applied.
        """

        for particle in particles:
            particle.SumForce = particle.SumForce + (self.Acceleration * particle.Mass)

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


