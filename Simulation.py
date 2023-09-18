from Particle import Particle
from Forces import *


class Simulation:
    def __init__( self ):
        self.Particles   = []
        self.Forces      = []    
        # Store the ground plane
        self.Constraints = []
        
    def Update( self, dt ):        
        for particle in self.Particles:    
            particle.SumForce = np.array([0,0,0])  #-- Zero All Sums of Forces in each iteration
            
        for force in self.Forces:             #-- Accumulate Forces
            force.Apply(self.Particles)
            
        for particle in self.Particles:       #-- Symplectic Euler Integration
            if( particle.Mass == 0 ): continue

            acceleration = particle.SumForce * ( 1.0 / particle.Mass )
            particle.Velocity += acceleration * dt
            particle.Position += particle.Velocity * dt
            
        for constraint in self.Constraints:   #-- Apply Penalty Constraints
            constraint.Apply( )
            
    def KineticEnergy( self ):
        energy = 0.0
        for particle in self.Particles:
            energy += 0.5 * particle.Mass * particle.Velocity * particle.Velocity
        return energy
        
    def PotentialEnergy( self ):
        energy = 0.0
        for particle in self.Particles:
            energy += 9.8 * particle.Mass * particle.Position.Z
        return energy
        
    def Display( self ):
        #-- Geometry
        #--
        geometry = []        
        for particle in self.Particles:
            geometry += particle.Display( )
        
        #-- Messages
        #--
        ke = self.KineticEnergy( )
        pe = self.PotentialEnergy( )
        print( "Kinetic   {0}".format( ke      ) )
        print( "Potential {0}".format( pe      ) )
        print( "Total     {0}".format( ke + pe ) )
        
        return geometry
        
    def BouncingParticles( self ):
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
        ground = Ground( self.Particles, 0.5 )
        self.Constraints.append( ground )

# Thhe ground of the simulation
class GroundPlane:

    def __init__( self, particles, loss = 1.0 ):
        self.Particles = particles
        self.Loss = loss
        
    def Apply( self ):
        for particle in self.Particles:
            if( particle.Position.Z < 0 ):
                particle.Position.Z *= -1
                particle.Velocity.Z *= -1
                particle.Velocity *= self.Loss