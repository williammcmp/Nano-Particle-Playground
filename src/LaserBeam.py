import numpy as np
import pandas as pd

class PulsedLaserBeam:
    """
    A class to model the properties of a Gaussian beam produced by a pulsed laser. 
    All units are SI, meaning 300 KHz = 300e3 Hz
    
    Attributes:
        wavelength (float): The wavelength of the laser in meters (default 1064 nm).
        power (float): The average power of the laser in watts (default 1 W).
        pulse_rate (float): The pulse repetition frequency in Hz (default 300 kHz).
        pulse_duration (float): The duration of each pulse in seconds (default 100 fs).
        numerical_aperture (float): The numerical aperture of the laser (default 0.14).

    Methods:
        calculate_rayleigh_range(n1, n2): Calculates the Rayleigh range for two media with different refractive indices.
        calculate_absorption_coefficient(k): Calculates the absorption coefficient based on Beer's Law.
        calculate_focus_volume(n1, n2): Computes the focus volume in two different media.
        get_beam_statistics(n1, k1, n2): Placeholder method for future beam statistics calculations.
    """

    def __init__(self, wavelength=1064e-9, power=1, pulse_rate=300e3, pulse_duration=100e-13, numerical_aperture=0.14, beam_waist = 0):
        # All units are SI, meaning 300 KHz = 300e3 Hz
        self.wavelength = wavelength
        self.power = power
        self.pulse_rate = pulse_rate
        self.pulse_duration = pulse_duration
        self.numerical_aperture = numerical_aperture
        self.reflectanc_factor = 0.52 #TODO add this as a process to calcuate
        self.abs_threshold = 0.64

        # Internal calculation methods
        self.beam_waist = beam_waist
        
        if beam_waist == 0:
            self.beam_waist = self._calculate_beam_waist()

        self.focus_area = self._calculate_focus_area()
        self.energy_per_pulse = self._calculate_energy_per_pulse()
        self.power_per_pulse = self._calculate_power_per_pulse()
        self.intensity_per_pulse = self._calculate_intensity_per_pulse()
        self.peak_intensity = self._calculate_peak_intensity()

    def calculate_rayleigh_range(self, n1=1.003, n2=3.565):
        """
        Calculates the Rayleigh range for two different refractive indices.

        Rayligh Range z_0 = π * ω^2 * n / λ
        
        Parameters:
            n1 (float): Refractive index of the first medium (default is air at 1.003).
            n2 (float): Refractive index of the second medium (default is silicon at 632.6 nm).

        Returns:
            tuple: Rayleigh range values for the first and second media.
        """
        z_n1 = (np.pi * (self.beam_waist ** 2 )* n1) / self.wavelength
        z_n2 = z_n1/n2

        return z_n1, z_n2
    
    def calculate_absorption_coefficient(self, k=0.00024048):
        """
        Calculates the optical absorption coefficient based on Beer's Law.

        Beer's Law - α = 4kπ/λ
        k - complex refractive index of ablated medium
        
        Parameters:
            k (float): Complex refractive index of the medium (default 0.0046).
        
        Returns:
            float: Absorption coefficient.
        """
        # return 4 * k * np.pi / self.wavelength
        return 4387
    
    def calculate_focus_volume(self, n1=1.003, n2=3.5650):
        """
        Calculates the focus volume in two different media using Rayleigh range values.

        Ellipoid Volume in each medium = 4/3 * a * b * c  = 4/3 * ⍵_0^2 * z_air * 1/2
        
        Parameters:
            n1 (float): Refractive index of the first medium (default is air at 1.003).
            n2 (float): Refractive index of the second medium (default is silicon at 632.6 nm).

        Returns:
            tuple: Focus volume values for both media.
        """
        z_n1, z_n2 = self.calculate_rayleigh_range(n1, n2)

        v1 = 4/6 * self.beam_waist ** 2 * z_n1
        v2 = 4/6 * self.beam_waist ** 2 * z_n2

        return v1, v2
    
    def get_beam_statistics(self, n1=1.003, k1=0.00024048, n2=3.5650):
        """
        Gathers all the properties and calculated metrics of the beam into a DataFrame.
        
        Parameters:
            n1 (float): Refractive index of the first medium.
            k1 (float): Absorption coefficient of the first medium.
            n2 (float): Refractive index of the second medium.
        
        Returns:
            DataFrame: A DataFrame containing all attributes and calculated properties of the beam.
        """
        # Get Rayleigh ranges for default refractive indices
        rayleigh_range_n1, rayleigh_range_n2 = self.calculate_rayleigh_range()

        # Get absorption coefficient for default k value
        absorption_coefficient = self.calculate_absorption_coefficient()

        data = {
            "Beam Waist (µm)": [f'{self.beam_waist*1e6:.3g}'],
            "Focus Area (cm^2)": [f'{self.focus_area*1e4:.3g}'],
            "Energy Per Pulse (µJ)": [f'{self.energy_per_pulse*1e6:.3g}'],
            "Power Per Pulse (W)": [f'{self.power_per_pulse:.3g}'],
            "Intensity Per Pulse (W/cm^2)": [f'{self.intensity_per_pulse*1e-4:.3g}'],
            "Rayleigh Range in Silicon (mm)" : [f'{rayleigh_range_n2*1e3:.3g}'],
            "Reflectanc Factor":[f'{self.reflectanc_factor}'],
            "Abs Threshold" : [f'{self.abs_threshold}']
        }

        return pd.DataFrame(data).T

    def _calculate_beam_waist(self):
        """
        Calculates the beam waist based on wavelength and numerical aperture.
        ω_o = λ / (π * NA)

        """
        return self.wavelength / (np.pi * self.numerical_aperture)
    
    def _calculate_focus_area(self):
        """
        Calculates the focus area of the beam.
        π * ω_0^2
        """
        return np.pi * self.beam_waist ** 2
    
    def _calculate_energy_per_pulse(self):
        """
        Calculates the energy per pulse.
        J = Power / Frequency = power/pulse_rate
        """
        return self.power / self.pulse_rate
    
    def _calculate_power_per_pulse(self):
        """
        Calculates the power of each pulse.
        P = J / τ = Energy per pulse / pulse duration
        """
        return self.energy_per_pulse / self.pulse_duration
    
    def _calculate_intensity_per_pulse(self):
        """
        Calculates the intensity of each pulse.
         J / A = energy per_pulse / focuse area
        """
        return self.power_per_pulse / self.focus_area
    
    def _calculate_peak_intensity(self):
        '''
        Calculates the peak Intensity
        I_0 = 2P_per_pulse / 𝜋⍵_0^2
        '''
        return 2 * self.power_per_pulse / (np.pi * (self.beam_waist ** 2))


    


    def SetBeamWaist( self ):
        # Change the beam radius --> need to alter the intensity profile
        pass







        