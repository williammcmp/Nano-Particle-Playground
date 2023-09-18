import numpy as np


class Particle:
    def __init__( self, position, velocity, mass, forceSum = np.array([0,0,0])):
        self.Position = np.array(position)  # [x,y,z] (m)
        self.Velocity = np.array(velocity)  # [x,y,z] (m/s)
        self.SumForce = np.array(forceSum)  # [x,y,z] (N)
        self.Mass     = mass                # kg

    def Display( self ):
        return [self.Position]
    
