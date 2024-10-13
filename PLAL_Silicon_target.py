#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 12 2024

A scrpt that models the absoption of a single laser pulse into Silicon

@author: william.mcm.p
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# -------------------
# Defining the Pulsed laser beam object -> source found on https://github.com/williammcmp/Nano-Particle-Playground/blob/main/src/LaserBeam.py
# -------------------

class PulsedLaserBeam:
    """
    A class to model the properties of a Gaussian beam produced by a pulsed laser. 
    All units are SI, meaning 300 KHz = 300e3 Hz
    
    https://github.com/williammcmp/Nano-Particle-Playground/blob/main/src/LaserBeam.py
    
    Attributes:
        wavelength (float): The wavelength of the laser in meters (default 1064 nm).
        power (float): The average power of the laser in watts (default 1 W).
        pulse_rate (float): The pulse repetition frequency in Hz (default 300 kHz).
        pulse_duration (float): The duration of each pulse in seconds (default 100 fs).
        numerical_aperture (float): The numerical aperture of the laser (default 0.14).
        reflectanc_factor (float): The reflectance factor at the silicon-air interface (default 0.3).
        abs_threshold (float): The absorption threshold of the material.
        beam_waist (float): The waist (radius) of the Gaussian beam at its focus.
        focus_area (float): The area of the beam focus, calculated as π * beam_waist^2.
        energy_per_pulse (float): The energy delivered by each laser pulse, calculated as power / pulse_rate.
        power_per_pulse (float): The power delivered by each pulse, calculated as energy_per_pulse / pulse_duration.
        intensity_per_pulse (float): The intensity of each pulse, calculated as power_per_pulse / focus_area.
        peak_intensity (float): The peak intensity at the center of the beam, calculated as 2 * power_per_pulse / (π * beam_waist^2).
        energy_density (float): The energy density at the focus, calculated as energy_per_pulse / focus_area.

    Methods:
        calculate_rayleigh_range(n1, n2): Calculates the Rayleigh range for two media with different refractive indices.
        calculate_absorption_coefficient(k): Calculates the absorption coefficient based on Beer's Law.
        calculate_focus_volume(n1, n2): Computes the focus volume in two different media.
        get_beam_statistics(n1, k1, n2): Placeholder method for future beam statistics calculations.
    """

    def __init__(self, wavelength=1064e-9, power=1, pulse_rate=300e3, pulse_duration=100e-13, numerical_aperture=0.14, beam_waist = 0):
        # All units are SI, meaning 300 KHz = 300e3 Hz
        # https://www.pveducation.org/pvcdrom/materials/optical-properties-of-silicon
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

        self.update()

    def update(self):
        """
        Re-calcuates the beams internal properties, for the case a prameter is changed after initislation
        """
        self.focus_area = self._calculate_focus_area()
        self.energy_per_pulse = self._calculate_energy_per_pulse()
        self.power_per_pulse = self._calculate_power_per_pulse()
        self.intensity_per_pulse = self._calculate_intensity_per_pulse()
        self.peak_intensity = self._calculate_peak_intensity()
        self.energy_density = self._calculate_energy_density()


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
    
    def calculate_absorption_coefficient(self, k=0.000024048):
        """
        Calculates the optical absorption coefficient based on Beer's Law.

        Beer's Law - α = 2kπ⍵/c where ⍵ is the angular frequencey of light
        k - complex refractive index of ablated medium

        
        Parameters:
            k (float): Complex refractive index of the medium (default 0.0046).
        
        Returns:
            float: Absorption coefficient.
        """
        # return (4 * k * np.pi / (self.wavelength) ) * 1e2
        # return 4387
        return 438700
    
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
        
        flunence = self.energy_per_pulse / self.focus_area

        data = {
            "Beam Waist (µm)": [f'{self.beam_waist*1e6:.3g}'],
            "Focus Area (cm^2)": [f'{self.focus_area*1e4:.3g}'],
            "Energy Per Pulse (µJ)": [f'{self.energy_per_pulse*1e6:.3g}'],
            "Fluence (J/cm^2)": [f'{flunence*1e-3:.3g}'],
            "Power Per Pulse (W)": [f'{self.power_per_pulse:.3g}'],
            "Intensity Per Pulse (W/cm^2)": [f'{self.intensity_per_pulse*1e-4:.3g}'],
            "Peak Intesnsity Per Pulse (W/cm^2)": [f'{self.peak_intensity*1e-4:.3g}'],
            "Rayleigh Range in Air (µm)" : [f'{rayleigh_range_n1*1e6:.3g}'],
            "Rayleigh Range in Silicon (µm)" : [f'{rayleigh_range_n2*1e6:.3g}'],
            "Absorption Coefficent ⍺ (m)" : [f'{absorption_coefficient:.3g}']
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
    
    def _calculate_energy_density(self, area = None):
        """
        Calculates the energy density of the beam in the transiant directionn at the z=0
        ρ = Energy per pulse / beams focus area
        """
        if area == None:
            area = self.focus_area
        return self.energy_per_pulse / area


    def SetBeamWaist(self, new_waist ):
        self.beam_waist =  new_waist
        
        self.update()

# -------------------
# Initialising the Pulsed laser beam object -> these can be changed
# -------------------

# Defining the Laser Beam's paramters and setting up the object
wavelength_m = 1030 * 1e-9  # Updated wavelength to 800 nm (in meters)
laser_power = 1 # Watts
pulse_rate = 100 * 1e3 # 100 kHz
pusle_duration = 100 * 1e-15 # 100 fs

numerical_aperture = 0.14 # Numerical apature

# defining the constants of silicon
silicon_heat_capacity_solid = 710 # J kg^-1 K^-1
silicon_heat_capacity_liquid = 910 # J kg^-1 K^-1
silicon_heat_capacity_gas = 910 # J kg^-1 K^-1 -> this is a best Guess

silicon_latent_heat_of_fussion = 1.97 * 1e6 # J kg^-1
silicon_latent_heat_of_vaporisation = 13.7 * 1e6 # J kg^-1

silicon_melting_temp = 1687 # K
silicon_boiling_temp = 2756 # K

silicon_density = 2330


Beam = PulsedLaserBeam(wavelength= wavelength_m,
                       power= laser_power,
                       pulse_rate=pulse_rate,
                       pulse_duration= pusle_duration,
                       numerical_aperture=numerical_aperture
                       )

z_air, z_silicon = Beam.calculate_rayleigh_range()

# Printing the Beams stats
print('Pulsed laser beam stats:\n',Beam.get_beam_statistics())

# -------------------
# Setting up required paramters for calculation -> feel free to change these
# -------------------

# The number of sub-divisions (or pixels) in the [x,y] space that the intensity profile will contain.
n_divisions = (255, 255) # -> the size of the voxel

# Getting energy at the [x,y] = 0 to show only the evolution energy into the silion wafer
r = np.linspace(-0.002 * np.sqrt(Beam.beam_waist), 0.002 * np.sqrt(Beam.beam_waist), n_divisions[0])
z = np.linspace(0, z_silicon * 2, n_divisions[1]) # Intensity decay into the medium W/cm^2

# Determain the spatical seperation between each slice of the energy profile
# Take the first element from the array diff as the depth of each segment.
# -> this is allowed because the z array is constructed with np.linspace and each element is evanly spaced
depth = np.diff(z)[0]
width = np.diff(r)[0]

# -------------------
# Energy density profile calculations
# -------------------

# Distribution the per pulse energy across the beams cross section in space
energy_into_silicon = np.exp((-2 * r ** 2 ) / (Beam.beam_waist) ** 2) * Beam.energy_density * (1 - Beam.reflectanc_factor) 

# Getting energy at the [x,y] = 0 to show only the evolution energy into the silion wafer
peak_energy = energy_into_silicon.max()

# setting up the intervals with a different step size on the z depth
Energy_into_silicon = np.abs(np.diff(peak_energy * np.exp(- Beam.calculate_absorption_coefficient() * z))) # Intensity decay accounting for complex refractive index

# Compute the energy profile
energy_profile = np.outer(energy_into_silicon , Energy_into_silicon).T

# -------------------
# Temp  profile calculations
# -------------------

# Calculate the volume for each slide in the z direction of the intensity gradient
# V = 𝜋 * r^2 * h
volume = depth * np.pi * Beam.beam_waist ** 2 

# Calcuate the mass of each voxel -> assuming each is the same size
voxel_mass = volume * silicon_density

# Convert energy density to energy per voxel
# Energy (J/m^2) * area (m^2) = energy
energy = energy_profile * np.pi * width ** 2 

#  Solid Limit -> E_{solid} = h_{fusion}m + mc_{solid}T_{melting}
solid_limit = silicon_latent_heat_of_fussion * voxel_mass + voxel_mass * silicon_heat_capacity_solid * silicon_melting_temp

# Liquid Limit ->  E_{liquid} = h_{vaporization}m + mc_{liquid}T_{boiling} + E_{solid}
liquid_limit = silicon_latent_heat_of_vaporisation * voxel_mass + voxel_mass * silicon_heat_capacity_liquid * silicon_boiling_temp + solid_limit

# From the limits above, determain which voxels are in each phase
# Soring the phase states in masks
solid = energy <= solid_limit
liquid =(energy > solid_limit) & (energy <= liquid_limit)
gas = energy > liquid_limit

# Calcuate the temps based on the masses
temperature_profile = np.zeros_like(energy)
# Solid phase temps
temperature_profile[solid] = energy[solid] / (voxel_mass * silicon_heat_capacity_solid)

# Liquid phase temps
temperature_profile[liquid] = (energy[liquid] - solid_limit) / (voxel_mass * silicon_heat_capacity_liquid)

# Gas phase temps
temperature_profile[gas] = (energy[gas] - liquid_limit) / (voxel_mass * silicon_heat_capacity_gas)

# Feedback on the temp profile of the material
print('Max temp = ', temperature_profile.max(), 'K')


# -------------------
# Plotting functions
# -------------------

# Energy profile plot
fig, ax = plt.subplots(1,2, figsize=(10, 6))

# Plot the image (heatmap)
img = ax[0].imshow(energy_profile, extent=[r.min(), r.max(), z.max(), z.min()], cmap='inferno', origin='upper')
fig.colorbar(img, ax=ax[0], shrink=0.75, label='Energy Density absorbed (J/$m^2$)')

# Label axes and set title
ax[0].set_xlabel('r (m)')
ax[0].set_ylabel('z (m) [depth of Silicon]')
ax[0].set_title("Absorbed Energy of Silicon")


# Temp profile plot
img = ax[1].imshow(temperature_profile, extent=[r.min(), r.max(), z.max(), z.min()], cmap='gist_heat', origin='upper')
fig.colorbar(img, ax=ax[1], shrink=0.75, label='Temperature (K)')

# Label axes and set title
ax[1].set_xlabel('r (m)')
ax[1].set_ylabel('z (m) [depth of Silicon]')
ax[1].set_title("Temperature profile of Silicon")

fig.tight_layout()
plt.show()


# The second figure showing the Phase Map of the Silicon
# Define a colormap for the phases (e.g., red for solid, green for liquid, blue for gas)
phase_cmap = ListedColormap(['red', 'green', 'blue'])
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Plot the temperature profile (heatmap) on the left
img0 = ax[0].imshow(temperature_profile, extent=[r.min(), r.max(), z.max(), z.min()], origin='upper')
fig.colorbar(img0, ax=ax[0], shrink=0.75, label='Temperature (K)')

# Add contour lines to the temperature profile for the different phases
contour = ax[0].contour(phase_map, levels=[0, 1, 2], colors=['red', 'green', 'blue'], 
                        extent=[r.min(), r.max(), z.max(), z.min()], origin='upper')
ax[0].clabel(contour, inline=True, fontsize=10, fmt={0: 'Melting', 1: 'Vaporisation'})

# Label axes and set title for the temperature profile
ax[0].set_xlabel('r (m)')
ax[0].set_ylabel('z (m) [depth of Silicon]')
ax[0].set_title("Temperature Profile of Silicon with Phase Contours")

# Plot the phase map (solid = red, liquid = green, gas = blue) on the right
img1 = ax[1].imshow(phase_map, extent=[r.min(), r.max(), z.max(), z.min()], origin='upper', cmap=phase_cmap)

# Add a colorbar for the phase map
cbar = fig.colorbar(img1, ax=ax[1], shrink=0.75, ticks=[0, 1, 2])
cbar.ax.set_yticklabels(['Solid', 'Liquid', 'Gas'])  # Label the colorbar ticks

# Label axes and set title for the phase map
ax[1].set_xlabel('r (m)')
ax[1].set_ylabel('z (m) [depth of Silicon]')
ax[1].set_title("Phase Map of Silicon")

# Adjust layout to prevent overlap
fig.tight_layout()
plt.show()
