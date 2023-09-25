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
    - Duration (float): A count of the simulation time used - accounts for separate or different time intervals

    Methods:
    - Update(dt): Updates the simulation for a given time step 'dt'.
    - Save(): Saves all particles' current position.
    - PlotPaths(title=""): Plots a 3D axis with the paths for each particle from the simulation.
    - Plot(): Plots a 2D axis with the position of the particles.
    - Run(duration=10, timeStep=0.1, saveHistory=True): Runs the particle simulation for a specified duration with a given time step.
    - Update(dt): Updates the simulation for a given time step 'dt'.
    - FroceList(): Returns a string listing information about the applied forces.
    """

    def __init__( self ):
        """
        Initializes a new Simulation instance.
        """

        self.Duration = 0

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

    def PlotPaths( self, title="" ):
        """
        Plots a 3D axis with the paths for each particle from the simulation.

        Parameters:
        - title (str): The title for the plot (optional).
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        colors = ['red', 'green', 'blue']

        for particle in self.Particles:
            ax.plot(particle.History[:, 0], particle.History[:, 1], particle.History[:, 2], c=colors[particle.Charge +1])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)

        plt.show()


    def Plot( self ):
        """
        Plots a 2d axis with the positon of the particles
        """
        title=f"{len(self.Particles)} Particles over {self.Duration}s"

        colors = ['red', 'green', 'blue']
        # Plot the data points
        for particle in self.Particles:
            plt.scatter(particle.Position[0], particle.Position[1], s=particle.Mass^2, c=colors[particle.Charge + 1])
        

        # Customize the plot (optional)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)
        # TODO add labels to indicate what color each charged particle is

        # Show the plot
        plt.show()


    def Run(self, duration=10, timeStep=0.1, saveHistory=True):
        """
        Run the particle simulation for a specified duration with a given time step.

        Parameters:
        - duration (float): The total duration of the simulation in seconds.
        - timeStep (float): The time step (seconds) at which the simulation is updated.
        - saveHistory (boolean): Controls if the positional history of the particles is saved.

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
        print("\nInitialising Particle Simulations.\nSetting up enviroment\n")

        self.Duration += duration # adding the sim time to track over multiple simulations


        print(f"\nSimulating particles:")
        startTime = time.time()


        # this saves re-evaluating if saveHistory over each iteration - faster compute time for larger iteration count
        if saveHistory:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Step"):
                self.Save() # saves the particles postion
                self.Update(timeStep)
        else:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Step"):
                self.Update(timeStep)


        print("\n Simulation:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {time.time() - startTime}s")
        print(f"\tTotal number of calculatios = {int(duration / timeStep) * len(self.Particles)}")

        print(f"\nForces:")
        print(self.FroceList())

    # TODO remove the particles from active list once that have become stationary -> np.diff(last 5 position) = 0.005?? may need to adjust the tollarance
    def Update( self, dt):    
        """
        Update the simulation for a given time step 'dt'.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """ 
        for particle in self.Particles:
            particle.SumForce = np.array([0,0,0])      

        for force in self.Forces:             #-- Accumulate Forces
            force.Apply(self.Particles)
            
        for particle in self.Particles:       #-- Symplectic Euler Integration
            if( particle.Mass == 0 ): continue

            acceleration = particle.SumForce * ( 1.0 / particle.Mass )
            particle.Velocity = particle.Velocity + (acceleration * dt) # v = u + at
            particle.Position = particle.Position + (particle.Velocity * dt) - 0.5 * acceleration * dt * dt # x = x_i + vt - 0.5at^2
            
        for constraint in self.Constraints:   #-- Apply Penalty Constraints
            constraint.Apply( )
    
    def FroceList( self ):
        """
        Returns a string listing information about the applied forces.

        Returns:
        - str: A string listing information about the applied forces.
        """
        forceList = ""
        for force in self.Forces:
            forceList += force.Info()
        
        return forceList
            