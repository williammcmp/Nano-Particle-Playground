import pytest
import numpy as np
from src.LaserBeam import PulsedLaserBeam

@pytest.fixture
def laser_beam():
    """Fixture to create a PulsedLaserBeam object with default parameters."""
    return PulsedLaserBeam(800e-9, 1, 80e6, 100e-13, 0.01)

def test_beam_waist(laser_beam):
    """Test the calculation of the beam waist."""
    expected_waist = 2.54e-5 # (m)
    assert np.isclose(laser_beam.beam_waist, expected_waist, atol=1e-7)

def test_focus_area(laser_beam):
    """Test the calculation of the focus area."""
    expected_focus_area = 20.37e-10 # (m^2)
    assert np.isclose(laser_beam.focus_area, expected_focus_area, atol=1e-12)

def test_energy_per_pulse(laser_beam):
    """Test the calculation of energy per pulse."""
    expected_energy = 1.25e-8 # (J)
    # expected_energy = 1/80e6 # (J)
    assert np.isclose(laser_beam.energy_per_pulse, expected_energy, atol=1e-7)

def test_intensity_per_pulse(laser_beam):
    """Test the calculation of intensity per pulse."""
    expected_intensity = 6.1359e11 # (W/m^2)
    # expected_intensity = 1.25e3/20.37e-10
    assert np.isclose(laser_beam.intensity_per_pulse, expected_intensity, atol=1e7)

def test_power_per_pulse(laser_beam):
    """Test the calculation of power per pulse."""
    expected_power = 1.25e3 # (w/m^2)
    # expected_power = (1/80e6)/100e-13 # (w/m^2)

    assert np.isclose(laser_beam.power_per_pulse, expected_power, atol=1e-9)

def test_rayleigh_range(laser_beam):
    """Test Rayleigh range calculation."""
    expected_range_n1 = (np.pi * (2.54e-5 ** 2) * 1.003) / 800e-9
    expected_range_n2 = expected_range_n1 / 3.881631
    rayleigh_range_n1, rayleigh_range_n2 = laser_beam.calculate_rayleigh_range()
    assert np.isclose(rayleigh_range_n1, expected_range_n1, atol=1e-4)
    assert np.isclose(rayleigh_range_n2, expected_range_n2, atol=1e-4)

def test_absorption_coefficient(laser_beam):
    """Test absorption coefficient calculation."""
    expected_absorption_coefficient = 4 * 0.0046 * np.pi / 800e-9
    absorption_coefficient = laser_beam.calculate_absorption_coefficient()
    assert np.isclose(absorption_coefficient, expected_absorption_coefficient, atol=1e-5)

def test_get_beam_statistics(laser_beam):
    """Test the DataFrame output of get_beam_statistics."""
    df = laser_beam.get_beam_statistics()
    assert df.loc[0, 'Wavelength (nm)'] == 800
    assert df.loc[0, 'Power (W)'] == 1
    # Additional checks for other fields can be included here
