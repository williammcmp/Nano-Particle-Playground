import pytest
import numpy as np
from src.Simulation import Simulation
from src.Particle import Particle
from src.Forces import *
from src.ParticleGenerator import GenerateTestParticles

@pytest.fixture
def simulation_fixture():
    sim = Simulation()
    return sim

# Test cases

def test_simulation_initialization(simulation_fixture):
    assert isinstance(simulation_fixture, Simulation)
    assert len(simulation_fixture.Particles) == 0
    assert len(simulation_fixture.Forces) == 0
    assert simulation_fixture.Duration == 0

def test_simulation_save(simulation_fixture):
    # Add a mock particle to the simulation
    mock_particle = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)
    simulation_fixture.Particles.append(mock_particle)

    # Call the Save method
    simulation_fixture.Save()

    # Check if the particle's history has been saved
    assert len(mock_particle.History) == 2  # Initial position + saved position

def test_add_particle(simulation_fixture):
    # Create mock particles
    mock_particle1 = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)

    # Add particles to the simulation
    simulation_fixture.AddParticles([mock_particle1])

    # Check if particles were added correctly
    assert len(simulation_fixture.Particles) == 1
    assert simulation_fixture.Particles[0] == mock_particle1

def test_add_multi_particles(simulation_fixture):
    # Create mock particles
    mock_particle1 = Particle([0, 0, 0], [0, 0, 0], 1.0, 0)
    mock_particle2 = Particle([1, 1, 1], [1, 1, 1], 2.0, 1)

    # Add particles to the simulation
    simulation_fixture.AddParticles([mock_particle1, mock_particle2])

    # Check if particles were added correctly
    assert len(simulation_fixture.Particles) == 2
    assert simulation_fixture.Particles[0] == mock_particle1
    assert simulation_fixture.Particles[1] == mock_particle2

def test_add_force(simulation_fixture):
    # Create mock forces
    mock_force1 = Gravity()

    # Add forces to the simulation
    simulation_fixture.AddForce([mock_force1])

    # Check if forces were added correctly
    assert len(simulation_fixture.Forces) == 1
    assert simulation_fixture.Forces[0] == mock_force1

def test_add_multi_forces(simulation_fixture):
    # Create mock forces
    mock_force1 = Gravity()
    mock_force2 = Gravity()

    # Add forces to the simulation
    simulation_fixture.AddForce([mock_force1, mock_force2])

    # Check if forces were added correctly
    assert len(simulation_fixture.Forces) == 2
    assert simulation_fixture.Forces[0] == mock_force1
    assert simulation_fixture.Forces[1] == mock_force2

def test_run(simulation_fixture):
    mock_force = Gravity()
    mock_barrier = Barrier()

    simulation_fixture.AddForce([mock_force, mock_barrier])


    # Adds the test particles to the sim
    GenerateTestParticles(simulation_fixture)

    assert len(simulation_fixture.Particles) == 3

    simulation_fixture.Run(5, 0.01, False)

    # for particle in simulation_fixture.Particles:
    #     # checks that the simulation puts the particles roughly where we expect
    #     assert np.allclose(particle.Position, np.array([0., 0.62, 0.001]))
