import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

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

    def PlotPaths( self ):
        """
        Plots a 3d axis with the paths for each particle from the simulation
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        for particle in self.Particles:
            ax.plot(particle.History[:, 0], particle.History[:, 1], particle.History[:, 2], s=particle.Mass)

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')

        plt.show()
        self.Display()

    def Plot( self , title=""):
        """
        Plots a 3d axis with the positon of the particles
        """
        # Create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the data points
        for particle in self.Particles:
            ax.scatter(particle.Position[0], particle.Position[1], particle.Position[2], s=particle.Mass)

        # Customize the plot (optional)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        # Show the plot
        plt.show()


    def Run(self, duration=10, timeStep=0.1, saveHistory=True):
        """
        Run the particle simulation for a specified duration with a given time step.

        This method sets up the simulation environment and updates the simulation for the specified duration
        by repeatedly calling the `Update` method at regular time intervals.

        Parameters:
        - duration (float): The total duration of the simulation in seconds.
        - timeStep (float): The time step (seconds) at which the simulation is updated. Smaller values increase resolution
        - saveHistory (boolean): Controls if the positional history of the partiles are saved. (faster if not saved)
        Example:
        ```
        sim.Run(duration=10.0, timeStep=0.1)
        sim.Run(duration=10.0, timeStep=0.1, False)
        ```

        Output:
        The simulation progresses in time steps, and progress is displayed using a progress bar.

        After completion, the total simulated time and time step size are printed.

        Returns:
        None
        """
        print("\nSetting up enviroment\n")
        print(f"Running Particle Simulation\n")
        startTime = time.time()

        # this saves re-evaluating if saveHistory over each iteration - faster compute time for larger iteration count
        if saveHistory:
            for x in tqdm(range(int(duration / timeStep))):
                self.Update(timeStep)
        else:
            for x in tqdm(range(int(duration / timeStep))):
                self.save() # saves the particles postion
                self.Update(timeStep)


        print("\n Simulation:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {time.time() - startTime}s")

        print(f"\nForces:")
        print(self.FroceList())

    def Update( self, dt ):    
        """
        Update the simulation for a given time step 'dt'.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """ 

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
    
    def FroceList( self ):
        forceList = ""
        for force in self.Forces:
            forceList += force.Info()
        
        return forceList
            
    


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