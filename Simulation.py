import matplotlib.pyplot as plt
import numpy as np
import threading

from Particle import Particle
from Forces import *


class Simulation:
    """
    Represents a particle simulation environment.

    Attributes:
    - Particles (list): A list of Particle objects in the simulation.
    - Forces (list): A list of Force objects acting on the particles.
    - Constraints (list): A list of constraint objects affecting the particles.

    Methods:
    - Update(dt): Updates the simulation for a given time step 'dt'.
    - KineticEnergy(): Calculates and returns the kinetic energy of the system.
    - PotentialEnergy(): Calculates and returns the potential energy of the system.
    - Display(): Displays information about the simulation, including kinetic and potential energy.
    - BouncingParticles(): Initializes a set of bouncing particles with forces and constraints.

    """

    def __init__( self ):
        """
        Initializes a new Simulation instance.
        """

        self.Particles   = []
        self.Forces      = []    
        # Store the ground plane
        self.Constraints = []

        self.Constraints.append( GroundPlane(self.Particles, 0.5) )
        
    def Save( self ):
        """
        Saves all particles' current position, could expand to include velocity
        """
        for particle in self.Particles:
            particle.Save()

    def Plot( self ):
        """
        Plots a 3d axis with the paths for each particle from the simulation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for particle in self.Particles:
            ax.plot(particle.History[:, 0], particle.History[:, 1], particle.History[:, 2])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        # self.Display()
        plt.show()

    def Update( self, dt ):    
        """
        Update the simulation for a given time step 'dt'.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """ 

        self.Save()

        for particle in self.Particles:    
            particle.SumForce = np.array([0,0,0])  #-- Zero All Sums of Forces in each iteration
            
        for force in self.Forces:             #-- Accumulate Forces
            force.Apply(self.Particles)
            
        for particle in self.Particles:       #-- Symplectic Euler Integration
            if( particle.Mass == 0 ): continue

            acceleration = particle.SumForce * ( 1.0 / particle.Mass )
            particle.Velocity = particle.Velocity + (acceleration * dt) # v = u + at
            particle.Position = particle.Position + (particle.Velocity * dt) - 0.5 * acceleration * dt * dt # x = x_i + vt - 0.5at^2
            
        for constraint in self.Constraints:   #-- Apply Penalty Constraints
            constraint.Apply( )
    
    def UpdateMultiThread(self, dt):
        """
        Update the simulation for a given time step 'dt'.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """

        self.Save()

        # Create a list to hold threads
        threads = []

        # Define a thread function for updating each particle
        def update_particle(particle):
            particle.SumForce = np.array([0, 0, 0])  # Zero All Sums of Forces in each iteration

            for force in self.Forces:
                force.Apply([particle])

            if particle.Mass == 0:
                return

            acceleration = particle.SumForce * (1.0 / particle.Mass)
            particle.Velocity = particle.Velocity + (acceleration * dt)
            particle.Position = (
                particle.Position
                + particle.Velocity * dt
                - 0.5 * acceleration * dt * dt
            )

        # Create and start a thread for each particle
        for particle in self.Particles:
            thread = threading.Thread(target=update_particle, args=(particle,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to finish
        for thread in threads:
            thread.join()

        for constraint in self.Constraints:
            constraint.Apply()
            
    def KineticEnergy( self ):
        """
        Calculate and return the kinetic energy of the system.

        Returns:
        - numpy.ndarray: The kinetic energy of the system in [x, y, z] joules.
        """

        energy = np.array([0,0,0])
        for particle in self.Particles:
            energy = energy + 0.5 * particle.Mass * particle.Velocity * particle.Velocity
        return energy
        
    def PotentialEnergy( self ):
        """
        Calculate and return the potential energy of the system.

        Returns:
        - numpy.ndarray: The potential energy of the system in [x, y, z] joules.
        """

        energy = np.array([0,0,0])
        for particle in self.Particles:
            energy[2] = energy[2] + 9.8 * particle.Mass * particle.Position[2]
        return energy
        
    def Display( self ):
        """
        Display information about the simulation, including kinetic and potential energy.

        Returns:
        - list: A list of particle geometries.
        """

        #-- Geometry
        #--
        geometry = []        
        for particle in self.Particles:
            geometry.append(particle.Display())
        
        #-- Messages
        #--
        ke = self.KineticEnergy( )
        pe = self.PotentialEnergy( )
        print( "Kinetic   {0}J".format( ke      ) )
        print( "Potential {0}J".format( pe      ) )
        print( "Total     {0}J".format( ke + pe ) )
        
        return geometry
    

        
        
    def BouncingParticles( self ):
        """
        Initialize a set of bouncing particles with forces and constraints.
        """

        #-- A number of particles along X-Axis with increasing mass
        #--
        for index in range( 10 ): 
            particle = Particle( 
                Point3d( index, 0, 100 ), 
                Vector3d.Zero, index + 1 )
            self.Particles.append( particle )
        
        #-- Setup forces
        #--
        gravity = Gravity( self.Particles )
        self.Forces.append( gravity )
        
        drag = Damping( self.Particles, 0.1 )
        self.Forces.append( drag )
        
        #-- Ground constraint
        #--
        ground = GroundPlane( self.Particles, 0.5 )
        self.Constraints.append( ground )

# Thhe ground of the simulation
class GroundPlane:
    """
    Represents the ground of the simulation.

    Attributes:
    - Particles (list): A list of Particle objects affected by the ground.
    - Loss (float): A coefficient representing energy loss upon bouncing.

    Methods:
    - Apply(): Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
    """

    def __init__( self, particles, loss = 1.0 ):
        """
        Initializes a new GroundPlane instance.

        Parameters:
        - particles (list): A list of Particle objects affected by the ground.
        - loss (float, optional): A coefficient representing energy loss upon bouncing. Default is 1.0.
        """
        self.Particles = particles
        self.Loss = loss
        
    def Apply( self ):
        """
        Applies the ground constraint to particles by reversing their position and velocity if they penetrate the ground.
        """

        for particle in self.Particles:
            if( particle.Position[2] < 0 ):
                particle.Position[2] = particle.Position[2] * -1
                particle.Velocity[2] = particle.Velocity[2] * -1 * self.Loss