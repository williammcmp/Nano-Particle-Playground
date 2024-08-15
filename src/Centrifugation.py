import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def plot_centrifuge_pos(pos, size, offset):
    fig, ax = plt.subplots(figsize=(5,4))
    x1, y1 = [0, 0.1], [0.01, 0.01]
    x2, y2 = [0, 0.1], [-0.01, -0.01]
    x3, y3 = [0, 0], [-0.01, 0.01]

    ax.scatter(pos, offset, s=size * 1e9, alpha=0.8, color='green')

    ax.plot(x1, y1, x2, y2, x3, y3, color='red', linewidth=2)
    ax.set_ylim([-0.011, 0.011])
    ax.set_xlim([-0.01, 0.11])
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    return fig

def plot_size_distro(pos, size):
    fig, ax = plt.subplots(figsize=(5,4))
    
    # Define the size ranges
    small_mask = size < 50 * 1e-9
    medium_mask = (size >= 50 * 1e-9) & (size < 150 * 1e-9)
    large_mask = (size >= 150 * 1e-9) & (size < 250 * 1e-9)
    very_large_mask = size > 250 * 1e-9

    # Plot density plots for each size category along positions
    sns.kdeplot(pos[small_mask], ax=ax, label='Small (0-50 nm)', color='blue', linewidth=2, alpha=0.8)
    sns.kdeplot(pos[medium_mask], ax=ax, label='Medium (50-150 nm)', color='green', linewidth=2, alpha=0.8)
    sns.kdeplot(pos[large_mask], ax=ax, label='Large (150-250 nm)', color='red', linewidth=2, alpha=0.8)
    sns.kdeplot(pos[very_large_mask], ax=ax, label='Very Large (>250 nm)', color='red', linewidth=2, linestyle='--', alpha=0.8)

    ax.set_title('Distribution of Particle Sizes Along Positions')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Density')
    ax.legend()

    return fig

def cal_sedimentation(size, rho_particles = 2230, rho_liquid = 997, liquid_viscosity = 1e-3, angular_vel = 2000, arm_length = 0.1):
    sed_coefficient = ((2 * (size ** 2) * (rho_particles - rho_liquid)) / (9 * liquid_viscosity)) # s = (2r^2(ρ_s - ρ_w) / (p * liquid_viscosity)
    sed_rate = (angular_vel ** 2) * arm_length * sed_coefficient # ⍵^2 * r * s --> in cm/s

    return sed_coefficient, sed_rate

def plot_size_probability(size, probability, title='Size distribution'):
    # plot initial probability
    fig, ax = plt.subplots(figsize=(5,4))
    ax.plot(size*1e9, probability, color='Green')

    ax.set_xlim([0,250])
    ax.set_ylim([0,1.1])

    ax.set_xlabel("Particle Size - Radius (nm)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(title)

    return fig

def cal_remaining_percent(size, prob, time, p_density, l_density, l_viscosity, rpm, arm_length, length):
    angular_velocity = rpm * 2 * np.pi
    sed_coefficient, sed_rate = cal_sedimentation(size, 
                                              p_density, l_density, 
                                              l_viscosity, 
                                              angular_velocity, arm_length)

    # Calculates the remaining % of supernate 
    remaining_percent  = prob * ((length - (sed_rate * time))/length)

    # Sets all negative values to 0
    remaining_percent  = np.where(remaining_percent < 0, 0, remaining_percent)

    return remaining_percent

def cal_supernate_and_pallets(size, prob, time, p_density, l_density, l_viscosity, rpm, arm_length, length):
    angular_velocity = rpm * 2 * np.pi
    sed_coefficient, sed_rate = cal_sedimentation(size, 
                                              p_density, l_density, 
                                              l_viscosity, 
                                              angular_velocity, arm_length)

    # Calculates the remaining % of supernate 
    supernate  = prob * ((length - (sed_rate * time))/length)

    # Sets all negative values to 0
    supernate  = np.where(supernate < 0, 0, supernate)

    pallets = prob - supernate

#     data_dict = {
#     'size': f'{size * 1e9:.1f}nm',  # Converted to nanometers with one decimal place
#     'supernat_i': f'{prob:.2f}',
#     'supernat_f': f'{supernate:.2f}',
#     'pallets': f'{pallets:.2f}',
#     'sed_rate': f'{sed_rate *1e5:.2f}'
# }

#     print(data_dict)

    return supernate, pallets


def plot_remaining_percent(time, remaining_percent, title=f"Supernatant Remaining Over Time"):
    fig, ax = plt.subplots(figsize=(5,4))

    ax.plot(time, remaining_percent * 1e2, linewidth=2)

    ax.set_ylim([0,100])
    ax.set_xlim([0,time[-1]])

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Supernate Percentage (%)")
    ax.set_title(title)

    return fig

def plot_centrifuge_data(run1 : np.array, run2 : np.array, mask_limit: int = 0, bar_width : int = 2):               
    # Creating the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Define the width of each bar and an offset for each series
    offset = bar_width

    # Plot for run1 (Side-by-side Bars)
    mask1 = run1['Radii(nm)'] > mask_limit
    radii1 = run1['Radii(nm)'][mask1]

    ax1.bar(radii1 - 2*offset, run1['Raw'][mask1], width=bar_width, label='Raw', alpha=0.8)
    ax1.bar(radii1 - 1*offset, run1['1kp'][mask1], width=bar_width, label='1kp', alpha=0.8)
    ax1.bar(radii1 , run1['2kp'][mask1], width=bar_width, label='2kp', alpha=0.8)
    ax1.bar(radii1 + offset , run1['4kp'][mask1], width=bar_width, label='4kp', alpha=0.8)
    ax1.bar(radii1 + 2*offset, run1['4ks'][mask1], width=bar_width, label='4ks', alpha=0.8)

    ax1.set_title("Run1: SiNP size concentrations - Recorded")
    ax1.set_ylabel("Composition (%)")
    ax1.set_xlabel("Particle Radii (nm)")
    ax1.legend()

    # Plot for run2 (Side-by-side Bars)
    mask2 = run2['Radii(nm)'] > mask_limit
    radii2 = run2['Radii(nm)'][mask2]

    ax2.bar(radii2 - offset, run2['Raw'][mask2], width=bar_width, label='Raw', alpha=0.8)
    ax2.bar(radii2, run2['2kp'][mask2], width=bar_width, label='2kp', alpha=0.8)
    ax2.bar(radii2 + offset, run2['4kp'][mask2], width=bar_width, label='4kp', alpha=0.8)

    ax2.set_title("Run2: SiNP size concentrations - Recorded")
    ax2.set_ylabel("Composition (%)")
    ax2.set_xlabel("Particle Radii (nm)")
    ax2.legend()

    return fig