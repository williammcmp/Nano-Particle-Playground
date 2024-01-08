# tests/test_forces.py

import pytest
import numpy as np
from src.Forces import Force, Gravity, Damping, Magnetic, Electric, Barrier
from src.Particle import Particle

@pytest.fixture
def sample_particles():
    # Create a sample list of particles for testing
    particle1 = Particle([0, 0, 1], [0, 0, 0], 1.0, 0)
    particle2 = Particle([1, 1, 1], [1, 1, 1], 2.0, 1)
    return [particle1, particle2]

@pytest.fixture
def sample_barrier():
    # Create a sample Barrier for testing
    return Barrier(damping=1.0, plane=np.array([0.0, 0.0, 1.0]), offset=np.array([0.0, 0.0, 0.0]))

@pytest.fixture
def sample_particles_behind_barrier():
    # Create a sample list of particles behind the barrier for testing
    particle1 = Particle([0, 0, -1], [0, 0, 1], 1.0, 0)
    particle2 = Particle([1, 1, -1], [1, 1, 1], 2.0, 1)
    return [particle1, particle2]

@pytest.fixture
def sample_particles_in_front_of_barrier():
    # Create a sample list of particles in front of the barrier for testing
    particle1 = Particle([0, 0, 1], [0, 0, -1], 1.0, 0)
    particle2 = Particle([1, 1, 1], [1, 1, -1], 2.0, 1)
    return [particle1, particle2]

def test_gravity_force(sample_particles):
    gravity_force = Gravity(magnitude=9.8)
    gravity_force.Apply(sample_particles)

    # Assert that the gravitational force is applied to the particles
    for particle in sample_particles:
        assert np.array_equal(particle.SumForce, np.array([0.0, 0.0, -9.8 * particle.Mass]))

def test_damping_force(sample_particles):
    damping_force = Damping(magnitude=0.5)
    damping_force.Apply(sample_particles)

    # Assert that the damping force is applied to the particles
    for particle in sample_particles:
        print(particle.SumForce)
        assert np.array_equal(particle.SumForce, np.array([0, 0, 0.5]))

def test_magnetic_force(sample_particles):
    magnetic_force = Magnetic(magnitude=1.0, direction=np.array([1, 2, 3]))
    magnetic_force.Apply(sample_particles)

    # Assert that the magnetic force is applied to the particles
    for particle in sample_particles:
        charge_term = particle.Charge * (np.cross(particle.Velocity, magnetic_force.Field()))
        assert np.array_equal(particle.SumForce, charge_term)

def test_electric_force(sample_particles):
    electric_force = Electric(magnitude=1.0, direction=np.array([1, 0, 0]))
    electric_force.Apply(sample_particles)

    # Assert that the electric force is applied to the particles
    for particle in sample_particles:
        charge_term = particle.Charge * electric_force.Field()
        assert np.array_equal(particle.SumForce, charge_term)


def test_barrier_application_behind_barrier(sample_barrier, sample_particles_behind_barrier):
    sample_barrier.Apply(sample_particles_behind_barrier)

    # Assert that the barrier reverses the velocity of particles behind it
    for particle in sample_particles_behind_barrier:
        assert np.array_equal(particle.Velocity[2],  -1)

def test_barrier_no_application_in_front_of_barrier(sample_barrier, sample_particles_in_front_of_barrier):
    sample_barrier.Apply(sample_particles_in_front_of_barrier)

    print(sample_barrier.Field())

    # Assert that the barrier does not affect particles in front of it
    for particle in sample_particles_in_front_of_barrier:
        assert np.array_equal(particle.Velocity[2], -1)  # Assuming initial velocity is [0, 0, 1]

def test_barrier_field_calculation(sample_barrier):
    # Assert that the field calculation is correct
    assert np.array_equal(sample_barrier.Field(), np.array([0.0, 0.0, 1.0]))