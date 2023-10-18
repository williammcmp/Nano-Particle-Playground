# src/Force.py
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
    
    def NanoApply( self, particles):
        """
        Applies the gravitational force to a list of nano-particles.

        Parameters:
        - particles (list): A list of Particle objects to which gravity is applied.
        """
        for particle in particles:
            particle.SumForce = particle.SumForce + (self.Acceleration * particle.Mass * 1e9)
    
    def Field( self ):
        return self.Acceleration

    def Info(self):
        return f"{self.name} = {self.Acceleration}"
    
    def __str__(self) :
        return "Gravity"

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

    def Field (self ):
        return np.array([self.Scaling, self.Scaling, self.Scaling])
    
    def __str__(self):
        return "Damping"

class Lorentz:
    """
    Represents a Lorentz force acting on particles in a simulation.

    Attributes:
    - eField (numpy.ndarray): An array representing the electric field in the Lorentz force.
    - bField (numpy.ndarray): An array representing the magnetic field in the Lorentz force.
    - name (str): A name for the Lorentz force.

    Methods:
    - Apply(particles): Applies the Lorentz force to a list of particles.
    - Info(): Returns information about the electric and magnetic fields in the Lorentz force.

    """
    def __init__( self, bField=np.array([0.0,0.0,0.0]), eField=np.array([0.0,0.0,0.0])):
        """
        Initializes a new Lorentz force instance.

        Parameters:
        - bField (numpy.ndarray): An array representing the magnetic field in the Lorentz force.
        - eField (numpy.ndarray): An array representing the electric field in the Lorentz force.
        """
        self.eField = eField
        self.bField = bField
        self.name = "Lorentz"

    def Apply( self, particles):
        """
        Applies the Lorentz force to a list of particles.

        Parameters:
        - particles (list): A list of Particle objects to which the force is applied.
        """

        for particle in particles:
            LorentzForce = particle.Charge * self.eField + particle.Charge *(np.cross(particle.Velocity, self.bField))
            particle.SumForce = particle.SumForce + (LorentzForce)
    

    def NanoApply( self, particles):
        """
        Applies the Lorentz force to a list of nano-particles.

        Parameters:
        - particles (list): A list of Particle objects to which the force is applied.
        """
        # TODO fix this for nano-scale
        for particle in particles:
            LorentzForce = particle.Charge * self.eField + particle.Charge*1e-19 *(np.cross(particle.Velocity, self.bField * (1e-9 ** 2)))
            particle.SumForce = particle.SumForce + (LorentzForce)

    def Field (self ):
        return [self.bField, self.eField]


    def Info( self ):
        """
        Returns information about the electric and magnetic fields in the Lorentz force.

        Returns:
        - str: A string containing information about the electric and magnetic fields.
        """
        return f"Magnetic = {self.bField}"

    def __str__(self):
        return "Lorentz"

# The ground of the simulation (Contraint)
class GroundPlane:
    """
    Represents the ground of the simulation.

    Attributes:
    - Particles (list): A list of Particle objects affected by the ground.
    - Loss (float): A coefficient representing energy loss upon bouncing.

    Methods:
    - Apply(): Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
    """

    def __init__( self, loss = 1.0 ):
        """
        Initializes a new GroundPlane instance.

        Parameters:
        - particles (list): A list of Particle objects affected by the ground.
        - loss (float, optional): A coefficient representing energy loss upon bouncing. Default is 1.0.
        """

        self.Loss = loss
        
    # try to apply the ground plane in the nano-update method(?)
    def Apply( self, particles):
        """
        Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
        """
        # Method not working for nano particles
        for particle in particles:

            if( particle.Position[2] < 0 ):
                particle.Position[2] = 0.000001 # reset the particle's position to above the ground plane
                # TODO add method to make the ground more sticky
                particle.Velocity = particle.Velocity * np.array([self.Loss, self.Loss, -1*self.Loss])


    def Info( self ):
        """
        Returns information about the ground plane
        Returns:
        - str: A string containing information about the ground plane
        """
        return f"Ground Plane energy loss factor = {np.array([self.Loss, self.Loss, -1*self.Loss])}"
    
    def __str__( self ):
        return "Ground Plane"
    

class Wall:
    """
    Represents the a wall in the simulation.

    Methods:
    - Apply(): Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
    """

    def __init__( self, loss = 1.0 ):
        """
        Initializes a new GroundPlane instance.

        Parameters:
        - particles (list): A list of Particle objects affected by the ground.
        - loss (float, optional): A coefficient representing energy loss upon bouncing. Default is 1.0.
        """

        self.Loss = loss
        
    # try to apply the ground plane in the nano-update method(?)
    def Apply( self, particles):
        """
        Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
        """
        # Method not working for nano particles
        for particle in particles:

            if( particle.Position[0] < 0 ):
                particle.Position[0] = 0.000001 # reset the particle's position to above the ground plane
                # TODO add method to make the ground more sticky
                particle.Velocity = particle.Velocity * np.array([-1 * self.Loss, self.Loss, self.Loss])


    def Info( self ):
        """
        Returns information about the ground plane
        Returns:
        - str: A string containing information about the ground plane
        """
        return f"Ground Plane energy loss factor = {np.array([-1 * self.Loss, self.Loss, -1*self.Loss])}"
    
    def __str__( self ):
        return "Wall"