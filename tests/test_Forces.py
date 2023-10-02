# test_forces.py

import pytest
import numpy as np
from src.Forces import Gravity, Damping, Lorentz, GroundPlane
from src.Particle import Particle

@pytest.fixture
def sample_particles():
    # Create a sample list of particles for testing
    particle1 = Particle([0, 0, 1], [0, 0, 0], 1.0, 0)
    particle2 = Particle([1, 1, 1], [1, 1, 1], 2.0, 1)
    return [particle1, particle2]

def test_gravity_force(sample_particles):
    gravity_force = Gravity(9.8)
    gravity_force.Apply(sample_particles)

    # Assert that the gravitational force is applied to the particles
    for particle in sample_particles:
        assert np.array_equal(particle.SumForce, np.array([0.0, 0.0, -9.8 * particle.Mass]))

def test_damping_force(sample_particles):
    damping_force = Damping(scaling=0.5)
    damping_force.Apply(sample_particles)

    # Assert that the damping force is applied to the particles
    for particle in sample_particles:
        assert np.array_equal(particle.SumForce, -0.5 * particle.Velocity)

def test_lorentz_force(sample_particles):
    e_field = np.array([1.0, 2.0, 3.0])
    b_field = np.array([4.0, 5.0, 6.0])

    lorentz_force = Lorentz(b_field, e_field)
    lorentz_force.Apply(sample_particles)

    # Assert that the Lorentz force is applied to the particles
    for particle in sample_particles:
        charge_term = particle.Charge * e_field + particle.Charge * (np.cross(particle.Velocity, b_field))
        assert np.array_equal(particle.SumForce, charge_term)

def test_ground_plane_constraint(sample_particles):
    ground_plane = GroundPlane(loss=0.5)
    ground_plane.Apply(sample_particles)

    # Assert that the ground constraint is applied to the particles
    for particle in sample_particles:
        if particle.Position[2] < 0:
            assert particle.Position[2] == 0.001
            assert np.array_equal(particle.Velocity, np.array([0, 0, 0]))

