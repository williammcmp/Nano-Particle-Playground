# src/Simulation.py
import matplotlib.pyplot as plt
import numpy as np
import time
import random
from tqdm import tqdm

from src.Forces import *


class Simulation:
    """
    Represents a particle simulation environment.

    Attributes:
    - Particles (list): A list of Particle objects in the simulation.
    - Forces (list): A list of Force objects acting on the particles.
    - Duration (float): A count of the simulation time used - accounts for separate or different time intervals

    Methods:
    - Update(dt): Updates the simulation for a given time step 'dt'.
    - Save(): Saves all particles' current position.
    - PlotPaths(title=""): Plots a 3D axis with the paths for each particle from the simulation.
    - Plot(): Plots a 2D axis with the position of the particles.
    - Run(duration=10, timeStep=0.1, saveHistory=True): Runs the particle simulation for a specified duration with a given time step.
    - Update(dt): Updates the simulation for a given time step 'dt'.
    - FroceList(): Returns a string listing information about the applied forces.
    - NanoRun(duration=10, timeStep=0.1, saveHistory=True): Runs the particle simulation with Brownian motion for a specified duration with a given time step.

    """

    def __init__(self):
        """
        Initializes a new Simulation instance.
        """

        self.Duration = 0

        self.Particles = []
        self.Forces = []

        # TODO update test to add the ground plane when needed
        # self.Constraints.append( GroundPlane(self.Particles, 0.5) )

    def AddParticles(self, Particles):
        """
        Adds list of Particles to the Simulation
        """
        for particle in Particles:
            self.Particles.append(particle)

    def AddForce(self, Forces):
        """
        Adds list of Forces to the Simulation
        """
        for force in Forces:
            self.Forces.append(force)

    def Save(self):
        """
        Saves all particles' current position, could expand to include velocity
        """
        for particle in self.Particles:
            particle.Save()

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
        computeTime, numCals
        """
        print("\nInitialising Particle Simulations.\nSetting up environment\n")

        self.Duration += duration  # adding the sim time to track over multiple simulations

        print(f"\nSimulating particles:")
        startTime = time.time()

        # this saves re-evaluating if saveHistory over each iteration - faster compute time for larger iteration count
        if saveHistory:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.Save()  # saves the particles position
                self.Update(timeStep)
        else:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.Update(timeStep)

        computeTime = time.time() - startTime
        numCals = int(duration / timeStep) * len(self.Particles) * (len(self.Forces) + 1)

        print("\n Simulation:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {computeTime}s")
        print(f"\tTotal number of calculations = {numCals}")

        print(f"\nForces:")
        # print(self.ForceList())

        return computeTime, numCals

    def NanoRun(self, duration=10, timeStep=0.1, saveHistory=True):
        """
        Run the particle simulation with Brownian motion for a specified duration with a given time step.

        Parameters:
        - duration (float): The total duration of the simulation in seconds.
        - timeStep (float): The time step (seconds) at which the simulation is updated.
        - saveHistory (boolean): Controls if the positional history of the particles is saved.

        Output:
        The simulation progresses in time steps, and progress is displayed using a progress bar.

        After completion, the total simulated time and time step size are printed.

        Returns:
        computeTime, numCals
        """
        print("\nInitialising Particle Simulations.\nSetting up environment\n")

        self.Duration += duration  # adding the sim time to track over multiple simulations

        print(f"\nSimulating particles with Brownian motion:")
        startTime = time.time()

        # this saves re-evaluating if saveHistory over each iteration - faster compute time for larger iteration count
        if saveHistory:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.Save()  # saves the particles position
                self.NanoUpdate(timeStep)
        else:
            for x in tqdm(range(int(duration / timeStep)), unit=" Time Steps"):
                self.NanoUpdate(timeStep)

        computeTime = time.time() - startTime
        numCals = int(duration / timeStep) * len(self.Particles) * (len(self.Forces) + 1)

        print("\n Simulation with Brownian motion:")
        print(f"\tParticles = {len(self.Particles)}\n\tSimulated time = {duration}s\n\tTime intervals = {timeStep}s\n\tCompute Time = {computeTime}s")
        print(f"\tTotal number of calculations = {numCals}")

        print(f"\nForces:")
        # print(self.ForceList())

        return computeTime, numCals

    def Update(self, dt):
        """
        Update the simulation for a given time step 'dt'.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """
        for particle in self.Particles:
            particle.SumForce = np.array([0, 0, 0])

        for force in self.Forces:  # -- Accumulate Forces and apply barrier constraints -> cuts down on number of loops completed intergating the barrier into the forces
            force.Apply(self.Particles)

        for particle in self.Particles:  # -- Calculate the position and velocities for each particle
            if particle.Mass == 0:
                continue

            acceleration = particle.SumForce * (1 / particle.Mass)  # a = F / m
            particle.Velocity = particle.Velocity + (acceleration * dt)  # v = u + at

            if particle.Velocity.all() != 0:  # no position update need for particles that are stationary
                particle.Position = particle.Position + (
                            particle.Velocity * dt) - 0.5 * acceleration * dt * dt  # x = x_i + vt - 0.5at^2


    def NanoUpdate(self, dt):
        """
        Update the simulation for a given time step 'dt'. Same as Update, but with Brownian motion.

        Parameters:
        - dt (float): The time step (seconds) for the simulation update.
        """
        for particle in self.Particles:
            particle.SumForce = np.array([0, 0, 0])

        for force in self.Forces:  # -- Accumulate Forces and apply barrier constraints
            force.Apply(self.Particles)

        for particle in self.Particles:  # -- Calculate the position and velocities for each particle
            if particle.Mass == 0:
                continue

            acceleration = particle.SumForce * (1 / particle.Mass)  # a = F / m
            particle.Velocity = particle.Velocity + (acceleration * dt)  # v = u + at

            # Brownian motion process
            brownian = (np.sqrt(dt) * np.random.normal(0, 1, 3)) * 0.5
            particle.Position = particle.Position + brownian  # applied the p(x,t) of Brownian motion

            if particle.Velocity.all() != 0:  # no position update needed for particles that are stationary
                particle.Position = particle.Position + (
                            particle.Velocity * dt) - 0.5 * acceleration * dt * dt  # x = x_i + vt - 0.5at^2



    def HasForce(self, forceName):
        """
        Checks if a force with the given name exists in the simulation.

        Parameters:
        - forceName (str): The name of the force to check.

        Returns:
        - bool: True if the force exists, False otherwise.
        """
        for force in self.Forces:
            if forceName == force.Name:
                return True

        return False

    def GetForce(self, forceName):
        """
        Returns a list of forces with the given name.

        Parameters:
        - forceName (str): The name of the force to retrieve.

        Returns:
        - list: A list of Force objects with the specified name.
        """
        forceList = []

        for force in self.Forces:
            if forceName == force.Name:
                forceList.append(force)

        return forceList
    
    def StreamletData( self ):
        
        position, velocity, force, mass, charge = self.__calNumPyArray()
                   
        return position, velocity, force, mass, charge       
            

    # converts Particle objs to array for after computing
    def __calNumPyArray(self):
        position = np.zeros([len(self.Particles), 3])
        velocity = np.zeros([len(self.Particles), 3])
        force = np.zeros([len(self.Particles), 3])
        mass = np.zeros([len(self.Particles), 1])
        charge = np.zeros([len(self.Particles), 1])

        print("\nobj -> array")
        for x in tqdm(range(len(self.Particles))):
            position[x] = self.Particles[x].Position
            velocity[x] = self.Particles[x].Velocity
            force[x] = self.Particles[x].SumForce
            mass[x] = self.Particles[x].Mass
            charge[x] = self.Particles[x].Charge

        return [position, velocity, force, mass, charge]