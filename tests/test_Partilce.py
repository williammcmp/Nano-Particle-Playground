#  Test for the Partlce class

import numpy as np

from src.Particle import Particle


# Sample test data
initial_position = [1.0, 2.0, 3.0]
initial_velocity = [0.1, 0.2, 0.3]
mass = 0.5
charge = 1.0
initial_force_sum = np.array([0.0, 0.0, 0.0])

def test_particle_initialization():
    particle = Particle(initial_position, initial_velocity, mass, charge, initial_force_sum)

    assert np.array_equal(particle.Position, np.array(initial_position))
    assert np.array_equal(particle.Velocity, np.array(initial_velocity))
    assert np.array_equal(particle.SumForce, np.array(initial_force_sum))
    assert particle.Mass == mass
    assert particle.Charge == charge
    assert np.array_equal(particle.History, np.array(initial_position))

def test_particle_save_history():
    particle = Particle(initial_position, initial_velocity, mass, charge, initial_force_sum)

    # Save history and check if the history has been updated
    new_position = [4.0, 5.0, 6.0]
    particle.Position = np.array(new_position)
    particle.Save()

    expected_history = np.vstack((initial_position, np.array(new_position)))

    assert np.array_equal(particle.History, expected_history)

def test_particle_display():
    particle = Particle(initial_position, initial_velocity, mass, charge, initial_force_sum)

    # Display the initial position
    assert np.array_equal(particle.Display(), np.array(initial_position))

    # Change the position and test again
    new_position = [7.0, 8.0, 9.0]
    particle.Position = np.array(new_position)
    assert np.array_equal(particle.Display(), np.array(new_position))

