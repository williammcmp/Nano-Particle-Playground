# test_simulation.py

import pytest

from src.Simulation import Simulation
from src.Particle import Particle
from src.Forces import *
from src.ParticleGenerator import *


@pytest.fixture
def simulation():
    return Simulation()

def test_simulation_initialization(simulation):
    assert isinstance(simulation, Simulation)
    assert len(simulation.Particles) == 0
    assert len(simulation.Forces) == 0
    assert len(simulation.Constraints) == 1  # Default constraint (GroundPlane)
    assert simulation.Duration == 0

def test_simulation_save(simulation):
    # Add a mock particle to the simulation
    mock_particle = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)
    simulation.Particles.append(mock_particle)

    # Call the Save method
    simulation.Save()

    # Check if the particle's history has been saved
    assert len(mock_particle.History) == 2  # Initial position + saved position

def test_add_particle(simulation):
    # Create mock particles
    mock_particle1 = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)


    # Add particles to the simulation
    simulation.AddParticles([mock_particle1])

    # Check if particles were added correctly
    assert len(simulation.Particles) == 1
    assert simulation.Particles[0] == mock_particle1


def test_add_multi_particles(simulation):
    # Create mock particles
    mock_particle1 = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)
    mock_particle2 = Particle([1, 1, 1], [1, 1, 1], 2.0, 1)

    # Add particles to the simulation
    simulation.AddParticles([mock_particle1, mock_particle2])

    # Check if particles were added correctly
    assert len(simulation.Particles) == 2
    assert simulation.Particles[0] == mock_particle1
    assert simulation.Particles[1] == mock_particle2

def test_add_force(simulation):
    # Create mock forces
    mock_force1 = Gravity()


    # Add forces to the simulation
    simulation.AddForce([mock_force1])

    # Check if forces were added correctly
    assert len(simulation.Forces) == 1
    assert simulation.Forces[0] == mock_force1


def test_add_multi_forces(simulation):
    # Create mock forces
    mock_force1 = Gravity()
    mock_force2 = Gravity()

    # Add forces to the simulation
    simulation.AddForce([mock_force1, mock_force2])

    # Check if forces were added correctly
    assert len(simulation.Forces) == 2
    assert simulation.Forces[0] == mock_force1
    assert simulation.Forces[1] == mock_force2

def test_run(simulation):
    mock_force = Gravity()

    simulation.AddForce([mock_force])

    # Adds the test the test particles to the sim
    GenerateTestParticles(simulation)

    assert len(simulation.Particles) == 3

    simulation.Run(5, 0.01, False)

    for particle in simulation.Particles:
        # checks that the simulation puts the particles roughly where we expect
        assert np.allclose( particle.Position, np.array([0.,0.62,0.001]))






    


# Mock Particle class for testing
class MockParticle(Particle):
    def __init__(self):
        super().__init__([0, 0, 0], [0, 0, 0], 1.0, 0)
