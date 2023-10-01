import matplotlib.pyplot as plt
import numpy as np
import time
from tqdm import tqdm

from src.Forces import *


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
        
    def AddParticles( self , Particles):
        """
        Adds Particles to the Simulation
        """
        for partilce in Particles:
            self.Particles.append(partilce)

    def AddForce( self , Forces ):
        """
        Adds Froces to the Simulation
        """
        for force in Forces:
            self.Forces.append(force)

    def Save( self ):
        """
        Saves all particles' current position, could expand to include velocity
        """
        for particle in self.Particles:
            particle.Save()

    def PlotPaths( self ):
        """
        Plots a 3D axis with the paths for each particle from the simulation.
        """
        title=f"{len(self.Particles)} Particles over {self.Duration}s"

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')


        for particle in self.Particles:
            ax.plot(particle.History[:, 0], particle.History[:, 1], particle.History[:, 2])

        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title(title)


        plt.show()

    # TODO add sim figures to plot (color legend, forces, particles mass range....)
    def Plot( self ):
        """
        Plots a 2d axis with the positon of the particles
        """
        title=f"{len(self.Particles):,} Particles over {self.Duration}s"

        [position, velocity, force, mass, charge] = self.__calNumPyArray(self.Particles)

        # cmap = ListedColormap(['red', 'green', 'blue'])  # Define your custom colormap here

        # Normalize the charge values to match the colormap indices
        # normalize = plt.Normalize(charge.min(), charge.max())
        normalize = plt.Normalize(mass.min(), mass.max())

        # Create a scatter plot with the custom colormap
        plt.scatter(
            position[:, 0],
            position[:, 1],
            # s=mass,
            c=mass,  # Use the charge values for color mapping
            # cmap="Set1_r",
            norm=normalize,
)
        # Customize the plot (optional)
        plt.xlabel('X (m)')
        plt.ylabel('Y (m)')
        plt.title(title)
        # TODO add labels to indicate what color each charged particle is
        plt.colorbar(label='Mass (kg)')

        # Show the plot
        plt.show()

    def Histogram(self):
        """
        Plots a histogram of particle's displacment from the origin.
        """
        title = f"{len(self.Particles):,} Particles over {self.Duration}s"

        position, _, _, mass, _ = self.__calNumPyArray(self.Particles)

        # Create a histogram of particle masses
        plt.hist(np.linalg.norm(position, axis=1), bins=20, edgecolor='k', alpha=0.7, color='blue')

        # Customize the plot (optional)
        plt.xlabel('Distance from orgin')
        plt.ylabel('Frequency')
        plt.title(title)

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
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.Save() # saves the particles postion
                self.Update(timeStep)
        else:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.Update(timeStep)


        print("\n Simulation:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {time.time() - startTime}s")
        print(f"\tTotal number of calculatios = {int(duration / timeStep) * len(self.Particles)}")

        print(f"\nForces:")
        print(self.FroceList())

    # faster run method - intended for very large number of particles
    def FastRun( self, duration=10, timeStep=0.1):
        print("\nInitialising Particle Simulations.\n\n\t...Setting up enviroment for FAST MODE\n")
        
        self.Duration += duration

        [position, velocity, force, mass, charge] = self.__calNumPyArray(self.Particles)


        print("\nSimulating particles (fast mode):")
        startTime = time.time()

        for x in tqdm(range(int(duration / timeStep))): # run the simulation FAST
            self.FastUpdate(0.01, position, velocity, force, mass, charge)

        endTime = time.time()

        self.__reloadParticels(self.Particles, position, velocity, force, mass, charge)

        print("\n Simulation:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {endTime - startTime}s")
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
            
        for particle in self.Particles:       #-- Cal the position and velocities for each particle
            if( particle.Mass == 0 ): continue

            acceleration = particle.SumForce * ( 1.0 / particle.Mass )
            particle.Velocity = particle.Velocity + (acceleration * dt) # v = u + at
            particle.Position = particle.Position + (particle.Velocity * dt) - 0.5 * acceleration * dt * dt # x = x_i + vt - 0.5at^2
            
        for constraint in self.Constraints:   #-- Apply Penalty Constraints
            constraint.Apply( )

    # faster update method, not as accurate and unable to store history (no path plot) - intedned for very large particle counts
    def FastUpdate(self, dt, position, velocity, force, mass, charge):
        # Cal forces
        force *= 0 # zero out forces, allows for re-calcuations each loop
        force += np.array([0,0,-9.8]) #gravity
        force += (charge * np.cross(velocity, np.array([10, -1, -2]))) #magnetic force
        
        # update position and velocitioes
        acceleration = force / mass
        velocity += acceleration * dt
        position += velocity*dt - 0.5*acceleration*dt*dt
        
        negative_z = position[:, 2] < 0 # find when particle below xy plane (-z values)
        
        velocity[negative_z] *= np.array([0,0,0]) # bounce logic (flip z and reduce x,y Velcoties)
        
        position[:,2] = abs(position[:,2]) # the ground plane
    
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
            

    # converts Particle objs to array for after computing
    def __calNumPyArray(self, Particles):
        position = np.zeros([len(Particles), 3])
        velocity = np.zeros([len(Particles), 3])
        force = np.zeros([len(Particles), 3])
        mass = np.zeros([len(Particles), 1])
        charge = np.zeros([len(Particles), 1])

        print("\nobj -> array")
        for x in tqdm(range(len(Particles))):
            position[x] = Particles[x].Position
            velocity[x] = Particles[x].Velocity
            force[x] = Particles[x].SumForce
            mass[x] = Particles[x].Mass
            charge[x] = Particles[x].Charge

        return [position, velocity, force, mass, charge]

    # Saves the particle array state to Particle objects
    def __reloadParticels(self, Particles, position, velocity, force, mass, charge):
        print("\narray -> obj")
        for x in tqdm(range(len(Particles))):
            Particles[x].Position = position[x]
            Particles[x].Velocity = velocity[x]
            Particles[x].SumForce = force[x]
            Particles[x].Mass = mass[x][0]
            Particles[x].Charge = charge[x][0]
        