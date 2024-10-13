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

from src.LaserBeam import PulsedLaserBeam

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
# Create a combined phase array with distinct values for each phase
phase_map = np.zeros_like(energy_profile)
phase_map[solid == 1] = 0  # Solid phase: assign value 0
phase_map[liquid == 1] = 1  # Liquid phase: assign value 1
phase_map[gas == 1] = 2  # Gas phase: assign value 2

# Define a colormap for the phases (e.g., red for solid, green for liquid, blue for gas)
phase_cmap = ListedColormap(['red', 'green', 'blue'])
fig, ax = plt.subplots(1, 2, figsize=(10, 6))

# Plot the temperature profile (heatmap) on the left
img0 = ax[0].imshow(temperature_profile, extent=[r.min(), r.max(), z.max(), z.min()], cmap='gist_heat', origin='upper')
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
