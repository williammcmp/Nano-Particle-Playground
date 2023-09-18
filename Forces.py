import numpy as np

# force due to gravity
class Gravity:
    def __init__( self, acceleration = 9.8 ):

        self.Acceleration = np.array([ 0.0, 0.0, -acceleration ])

    # Applies the gravity acceleration into each particle
    def Apply( self, particles):
        for particle in particles:
            particle.SumForce = particle.SumForce + self.Acceleration * particle.Mass

# Viscous Drag Force
class Damping:
    def __init__( self, scaling = 1.0 ):
        self.Scaling   = scaling

    def Apply( self, particles ):
        for particle in particles:
            particle.SumForce += particle.Velocity * -self.Scaling


