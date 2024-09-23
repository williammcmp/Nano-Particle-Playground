#  Loading in the Libraries
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
# import seaborn as sns

# Loading in the Simulation Objects
from src.streamlitText import *
from src.Centrifugation import *


# ------------
# Display properties
# ------------
# Set page layout to wide
st.set_page_config(page_title="Nano Particle Simulation", 
                   initial_sidebar_state="collapsed",
                   page_icon="img/NPP-icon-blueBG.png")



# makes the plots in line with the style of the application dark mode
rc = {'figure.figsize':(6,5),

        'axes.facecolor':'#0e1117',
        'axes.edgecolor': '#0e1117',
        'axes.labelcolor': 'white',
        'figure.facecolor': '#0e1117',
        'patch.edgecolor': '#0e1117',
        'text.color': 'white',
        'xtick.color': 'white',
        'ytick.color': 'white',
        'grid.color': 'grey',
        'font.size' : 18,
        'axes.labelsize': 16,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16}
plt.rcParams.update(rc)

with st.sidebar:
    rho_particles = st.number_input(r"Density of the colloids ($$kg/m^2$$)", 500, 3000, 2330) # density of the particles used
    rho_liquid = st.number_input(r"Density of liquid ($$kg/m^2$$)", 50, 3000, 997) # default density 
    liquid_viscosity = st.number_input(r"Viscosity of liquid ($$m Pa.s$$)", 0.1, 2.0, 1.0) # default density iw water at 20C
    rpm = st.number_input(r"Centrifuge speed ($$RPM$$)", 1, 40000, 4000) # RPM of the centrifuge 
    arm_length = st.number_input(r"Centrifuge arm length ($$cm$$)", 1, 20, 10) * 1e-2
    length = st.number_input(r"Length of the container ($$cm$$)", 1, 20, 1) * 1e-2
    duration = st.number_input(r"Duration ($$min$$)", 1, 120, 10) # Duration of Centrifugation

st.header("Centrifugation of Colloids")

text_col, plot_col = st.columns([1,1])

with text_col:
    st.subheader("Supernatant over time")
    st.markdown(r"""During centrifugation, the percentage of supernatant decreases as particles sediment to the bottom of the container. The rate at which this occurs depends on the particle size, RPM, and the total amount of particles.

After the first centrifugation cycle, the remaining supernatant is redistributed across the container. In the second cycle, particles must travel a greater distance to sediment, leading to a shallower gradient as seen in the plot. This demonstrates how subsequent cycles affect the supernatant percentage, with each cycle requiring more time for the same amount of particle sedimentation.""")

with plot_col:
    # How does mutiple runs look for a single size particle?
    size = st.number_input('Particle Radius (nm)', 1, 250, 100) * 1e-9

    runs = 2 
    prob = 1

    fig, ax = plt.subplots()

    for run in range(runs):
        time = np.linspace(0,30,100)
        prob_remaining = cal_remaining_percent(size, prob, time,
                                                rho_particles, rho_liquid, liquid_viscosity,
                                                rpm, arm_length, length)
        prob = prob_remaining[-1]
        ax.plot(time, prob_remaining * 1e2, label=f"Cycle: {run +1}", alpha = 0.8, linewidth=4) 

        # Add a gray dotted line at y=prob
        ax.axhline(y=prob * 1e2, color='lightgray', linestyle='--', linewidth=2, alpha=0.5)

    ax.set_ylim([0,100])
    ax.set_xlim([0,time[-1]])

    ax.set_xlabel("Centrifugation Time (min)")
    ax.set_ylabel("Supernate Percentage (%)")
    ax.set_title(f'Supernatant Remaining')
    ax.legend()

    st.pyplot(fig)
    st.caption(f'Particle Radius: {size * 1e9:.0f}nm')



with st.expander("How does the ratio of compasition over time change?"):
    st.markdown(centrifugation_ratios())

st.divider()


text_col, plot_col = st.columns([1,1])

with text_col:
    st.subheader("Pallets over time")
    st.markdown(centrifugation_pallets())

with plot_col:
    # size1 = st.number_input('Particle Radius (nm)', 1, 250, 100) * 1e-9
    runs = st.number_input('Number of Runs', 1, 4, 2)
    prob = 1

    # Define a color cycle for different runs
    colors = ['red', 'blue', 'green', "purple"]

    fig, ax = plt.subplots()
    for run in range(runs):
        times = np.linspace(0,30,100)
        supernate, pallets = cal_supernate_and_pallets(size, prob, times,
                                                rho_particles, rho_liquid, liquid_viscosity,
                                                rpm, arm_length, length)
        prob = supernate[-1]
        ax.plot(times, supernate * 1e2, label=f"Supernatant: {run +1}", alpha = 0.8, linewidth=4, color=colors[run]) 
        ax.plot(times, pallets * 1e2, label=f"Pallets: {run + 1}", alpha=0.7, linewidth = 3, linestyle='--', color=colors[run])

        # Add a gray dotted line at y=prob
        ax.axhline(y=prob * 1e2, color='gray', linestyle='--', linewidth=1, alpha=0.5)

    ax.set_ylim([0,100])
    ax.set_xlim([0,times[-1]])

    ax.set_xlabel("Centrifugation Time (min)")
    ax.set_ylabel("Supernate Percentage (%)")
    ax.set_title(f'Colloid Composition')
    ax.legend()

    st.pyplot(fig)
    st.caption(f'Particle Radius: {size * 1e9:.0f}nm')

st.divider()

text_col, plot_col = st.columns([1,1])

with text_col:
    st.subheader("Effect of centrifugation speeds")
    # st.markdown()

with plot_col:
    rpms = np.array([4000, 10000, 20000])
    j = 0

    fig, ax = plt.subplots()

    colors = ['red', 'blue', 'green', 'purple', 'brown']

    for speed in rpms:
        prob = 1
        times = np.linspace(0, duration, 200)  # Replace `time` with `duration` to avoid overwriting
        prob, pallets = cal_supernate_and_pallets(size, prob, times,
                                                  rho_particles, rho_liquid, liquid_viscosity,
                                                  speed, arm_length, length)

        ax.plot(times, prob * 1e2, label=f"RPM: {speed * 1e-3:.0f}K", alpha=0.8, linewidth=2, color=colors[j])
        j += 1

    ax.set_ylim([0, 100])
    ax.set_xlim([0, duration])

    ax.set_xlabel("Centrifugation Time (min)")
    ax.set_ylabel("Supernate Percentage (%)")
    ax.set_title(f'Centrifuge Speeds')
    ax.legend()

    st.pyplot(fig)
    st.caption(f'Particle Radius: {size * 1e9:.0f} nm')


st.divider()

text_col, plot_col = st.columns([1,1])

with text_col:
    st.subheader("Overall Composition of the Colloid")
    # st.markdown()

with plot_col:
    # size1 = st.number_input('Particle Radius (nm)', 1, 250, 100) * 1e-9
    count = 100
    sizes = np.linspace(1,251,count) * 1e-9

    prob = np.ones(count)
    pallets = np.zeros(count)


    # Define a color cycle for different runs
    colors = ['red', 'blue', 'green', 'purple']

    fig, ax = plt.subplots()

    ax.plot(sizes*1e9, prob * 1e2, label=f"Inital state", linewidth=2)


    for j in range(runs):

        for i in range(count):
            prob[i], pallets[i] = cal_supernate_and_pallets(sizes[i], prob[i], duration,
                                                rho_particles, rho_liquid, liquid_viscosity,
                                                rpm, arm_length, length)
        

        ax.plot(sizes*1e9, prob * 1e2, label=f"Cycle: {j +1}", alpha = 0.8, linewidth=2, color=colors[j]) 
        ax.plot(sizes*1e9, pallets * 1e2, alpha=0.7, linestyle='--', color=colors[j])

    ax.set_xlabel("Particle Radius (nm)")
    ax.set_ylabel("Probability (%)")
    ax.set_title(f"Colloid Centrifuge")
    ax.legend()

    st.pyplot(fig)
    st.caption(f'Centrifugation Time: {duration}, Centrifugation Speed: {rpm *1e-3 :.0f}K')


