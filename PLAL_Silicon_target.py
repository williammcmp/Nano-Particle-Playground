#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Saturday October 12 2024

A scrpt that models the absoption of a single laser pulse into Silicon

@author: william.mcm.p
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# Constants
LaserPower = 5  # Watt
RepRate = 200000  # per second
PulseEnergy = LaserPower / RepRate
lambdan = 1030e-9 / 1.333  # lambda in water
ObjNA = 0.14
speciHSi = 710  # J/kgK
speciHSiMolten = 1010  # J/kgK
rhoSi = 2330  # kg/m^3
rhoSiMolten = 2520  # kg/m^3
w0 = 0.61 * lambdan / ObjNA
alphaSi = 4.387e5  # per m
AbsPulseE = PulseEnergy * 0.793  # Reflectance is 79.3% for water-silicon interface

# Functions
def AbsEperVol(rrr, zzz):
    return 2 * AbsPulseE * alphaSi / (np.pi * w0**2) * np.exp(-(2 * (rrr / w0)**2 + alphaSi * zzz))

def TempRise(rrr, zzz):
    return (2 * AbsPulseE * alphaSi / (np.pi * w0**2) * np.exp(-(2 * (rrr / w0)**2 + alphaSi * zzz)) -
            speciHSi * rhoSi * 1390 - 1787000 * rhoSiMolten -
            speciHSiMolten * rhoSiMolten * 945 - 13722000 * rhoSiMolten) / (speciHSi * rhoSi)

def TempRise1(rrr, zzz):
    return speciHSi * rhoSi * 1390

def TempRise2(rrr, zzz):
    return speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten

def TempRise3(rrr, zzz):
    return speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten + speciHSiMolten * rhoSiMolten * 945

def TempRise4(rrr, zzz):
    return speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten + speciHSiMolten * rhoSiMolten * 945 + 13722000 * rhoSiMolten

# Define the function VolTP to calculate volume based on temperature rise (TP)
def VolTP(TP):
    return (np.pi * w0**2) / alphaSi / 4 * (np.log(TP * (np.pi * w0**2) / alphaSi / 2 / AbsPulseE))**2


# Plotting AbsEperVol and TempRise as a function of rrr
r_values = np.linspace(0, 10*1e-6, 1000)  # r in meters
EperVol_values = [AbsEperVol(r, 0) for r in r_values]
TP1_values = [TempRise1(r, 0) for r in r_values]
TP2_values = [TempRise2(r, 0) for r in r_values]
TP3_values = [TempRise3(r, 0) for r in r_values]
TP4_values = [TempRise4(r, 0) for r in r_values]

plt.figure()
plt.plot(r_values * 1e6, EperVol_values, label='AbsEperVol')
plt.plot(r_values * 1e6, TP1_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP2_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP3_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP4_values, linestyle='-', color='gray')
plt.xlabel('Radius (µm)')
plt.ylabel('Energy ($J/m^3$)')
plt.title("Energy Profile at water-Si interface")

plt.show()

# Plotting AbsEperVol and TempRise as a function of zzz
z_values = np.linspace(0, 10e-6, 1000)  # z in meters
EperVol_values_z = [AbsEperVol(0, z) for z in z_values]
TP1_values_z = [TempRise1(0, z) for z in z_values]
TP2_values_z = [TempRise2(0, z) for z in z_values]
TP3_values_z = [TempRise3(0, z) for z in z_values]
TP4_values_z = [TempRise4(0, z) for z in z_values]

plt.figure()
plt.plot(z_values * 1e6, EperVol_values_z, label='AbsEperVol')
plt.plot(r_values * 1e6, TP1_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP2_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP3_values, linestyle='-', color='gray')
plt.plot(r_values * 1e6, TP4_values, linestyle='-', color='gray')
plt.xlabel('Depth (µm)')
plt.ylabel('Energy density ($J/m^3$)')
plt.title('Absorbed Energy Profile into Silicon')

plt.show()


# Creating data for contour plot
rr_values = np.arange(0, 4e-6, 0.01e-6)  # Radius values in meters
zz_values = np.arange(0, 4e-6, 0.01e-6)  # Depth values in meters
wq = np.array([[AbsEperVol(rr, zz) for zz in zz_values] for rr in rr_values])

# Define contour levels for temperature rise values
phase_energy_limits = [speciHSi * rhoSi * 1390, 
                  speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten, 
                  speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten + speciHSiMolten * rhoSiMolten * 945, 
                  speciHSi * rhoSi * 1390 + 1787000 * rhoSiMolten + speciHSiMolten * rhoSiMolten * 945 + 13722000 * rhoSiMolten]

# Convert mesh grid values to nm for labeling
X, Y = np.meshgrid(zz_values * 1e6, rr_values * 1e6)
# Energy profile plot
fig, ax = plt.subplots()

# Plot the image (heatmap)
img = ax.imshow(wq, extent=[0, 4, 0, 4], cmap='inferno', origin='lower', aspect='auto')
fig.colorbar(img, ax=ax, shrink=1, label='Energy Density Absorbed (J/m$^3$)')


# Add contour lines for phase transitions
contour = ax.contour(X, Y, wq, levels=phase_energy_limits, colors='white', linewidths=0.8)
ax.clabel(contour, inline=True, fontsize=10, fmt={phase_energy_limits[0]: 'Solid', 
                                                   phase_energy_limits[1]: 'Solid + Liquid', 
                                                   phase_energy_limits[2]: 'Liquid', 
                                                   phase_energy_limits[3]: 'Liquid + Gas'})

# Label axes and set title
ax.set_xlabel('Radius (10 nm)')
ax.set_ylabel('Depth (10 nm)')
ax.set_title("Absorbed Energy Profile of Silicon")

# Adjust layout for better fit
fig.tight_layout()
plt.show()

solid = wq <= phase_energy_limits[0]
solid_and_liquid = (wq > phase_energy_limits[0]) & (wq <= phase_energy_limits[1])
liquid = (wq > phase_energy_limits[1]) & (wq <= phase_energy_limits[2])
liquid_and_vapour = (wq > phase_energy_limits[1]) & (wq <= phase_energy_limits[2])
gas = wq > phase_energy_limits[2]

# Create a combined phase array with distinct values for each phase
phase_map = np.zeros_like(wq)
phase_map[solid == 1] = 0  # Solid phase: assign value 0
phase_map[solid_and_liquid == 1] = 1  # Solid and Liquid region: value 1
phase_map[liquid == 1] = 2  # Liquid phase: assign value 2
phase_map[liquid_and_vapour == 1] = 3  # Liquid and Vapor region: value 3
phase_map[gas == 1] = 4  # Gas phase: assign value 4

# Define a colormap for the phases (e.g., red for solid, orange for solid + liquid, etc.)
phase_cmap = ListedColormap(['red', 'orange', 'green', 'lightseagreen', 'blue'])

# Plot the phase map (solid = red, liquid = green, gas = blue) on the right
fig, ax = plt.subplots()
img = ax.imshow(phase_map, origin='lower', cmap=phase_cmap, extent=[0, 4, 0, 4], aspect='auto')

# Add a colorbar for the phase map with custom labels
cbar = fig.colorbar(img, ax=ax, shrink=1, ticks=[0, 1, 2, 3, 4])
cbar.ax.set_yticklabels(['Solid', 'Solid + Liquid', 'Liquid', 'Liquid + Gas', 'Gas'])  # Label the colorbar ticks

# Label axes and set title for the phase map
ax.set_xlabel('z depth (10 nm)')
ax.set_ylabel('radius (10 nm)')
ax.set_title("Phase Map of Silicon")

# Adjust layout to prevent overlap
fig.tight_layout()
plt.show()

# 3D Plot
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
X, Y = np.meshgrid(rr_values * 1e6, zz_values * 1e6)
ax.plot_surface(X, Y, wq * 1e-10, cmap='viridis', edgecolor='none')
ax.set_xlabel('Radius (um)')
ax.set_ylabel('Depth (um)')
ax.set_zlabel('Energy Density ($MJ/m^3$)')
ax.set_title('Energy Density')
fig.tight_layout()
plt.show()

# Calculate volumes for each transition
vol_solid_liquid = (VolTP(phase_energy_limits[0]) - VolTP(phase_energy_limits[1])) * 5.4e7
vol_liquid = (VolTP(phase_energy_limits[1]) - VolTP(phase_energy_limits[2])) * 5.4e7
vol_liquid_vapour = (VolTP(phase_energy_limits[2]) - VolTP(phase_energy_limits[3])) * 5.4e7
vol_vapour = VolTP(phase_energy_limits[3]) * 5.4e7
vol_tp1 = VolTP(phase_energy_limits[0]) * 5.4e7
vol_tp2 = VolTP(phase_energy_limits[1]) * 5.4e7
vol_tp3 = VolTP(phase_energy_limits[2]) * 5.4e7
vol_tp4 = VolTP(phase_energy_limits[3]) * 5.4e7

# Print the results in cubic micrometers (µm³)
print(f"Vol of Solid and Liquid = {vol_solid_liquid} µm³")
print(f"Vol of Lliquid = {vol_liquid} µm³")
print(f"Vol of Lliquid and Vapour = {vol_liquid_vapour} µm³")
print(f"Vol of Vapour = {vol_vapour} µm³")
print(f"Vol of TP1 = {vol_tp1} µm³")
print(f"Vol of TP2 = {vol_tp2} µm³")
print(f"Vol of TP3 = {vol_tp3} µm³")
print(f"Vol of TP4 = {vol_tp4} µm³")