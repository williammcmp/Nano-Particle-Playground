import streamlit as st

def how_to_use_info(simMode):
    intro_info = f'''
    ## How to Use the Simulation

    1. **Select Simulation Mode:** Choose your desired simulation mode from the sidebar. The current mode is `{simMode}`.
    2. **Adjust Simulation Settings:** Modify simulation settings in the sidebar as needed.
    3. **Real-Time Updates:** The simulation will dynamically update after each setting change.
    4. **Explore Results:** Dive into the results and visualizations displayed on the main page.

    ## Simulation Modes

    - **Silicon Nano-Particles:** Simulate the behavior of silicon nanoparticles under different magnetic fields to match selected experimental data and accounts for Brownian motion. In this mode, the primary simulation setting is the number of particles. For larger particle counts (>2,000), compute time may increase.
    - **Three Particle System:** This simplified mode illustrates the behavior of three particles. It's ideal for observing how individual particles behave under various forces.
    - **Standard:** Gain complete control over the simulation, particles, and forces.

    ## Particle Initial Distribution

    This section is available only in "Standard" Simulation mode.

    - **Starting Position:** Choose whether particles begin at the origin or have random positions. When using random positions, you can specify the average initial positions (X, Y, Z) following a normal distribution.
    - **Mass Range:** Define the mass range of the particles (uniform distribution).
    - **Average Initial Kinetic Energy:** Set the average initial kinetic energy from a normal distribution. Initial kinetic energy is used as a more defining parameter than an initial average velocity.
    - **Charged Particles:** Toggle charged particles (positive, negative, or neutral).

    ## Simulation Forces

    - **Gravity:** Apply standard gravity, defined as -9.8 m/s² along the z-axis.
    - **Magnetic Field:** Introduce a linear magnetic field to act on charged particles that are in motion (requires `charged particles`). You can define its axial strength.
    - **Electric Field:** Implement a linear electric field to act on charged particles (requires `charged particles`). You can specify its axial strength.

    ## Simulation Constraints

    - **Ground Plane:** Add a ground plane at z = 0. By default, when a particle collides with the ground, it comes to rest (sticky ground).
    - **Particle Bounce:** Enable particle bouncing (requires a `Ground Plane`). This removes the sticky ground effect.
    - **Bounce Factor:** Define the bounce factor to control the bounciness of the ground plane. A factor of:
        - < 1 signifies an inelastic collision (energy loss per collision).
        - = 1 represents an elastic collision (no energy gained or lost per collision).
        - \> 1 indicates a driven system (energy gained per collision).

    ## Running the Simulation on Your Own Machine

    For those seeking enhanced performance and a personalized experience, you can set up the simulation locally. Clone the code from the [Nano Particle Playground GitHub repository](https://github.com/williammcmp/Nano-Particle-Playground), where you'll find comprehensive instructions to guide you through the setup and execution process. This empowers you to customize the simulation to suit your specific preferences.
    '''
    return intro_info

def sim_intro():
    intro = f'''
    **Students:** Christ Nohan, William McMahon-Puce

    **Superviors:** James Chon, Saulius Juodkazis

    In an effort to characterize Silicon Nano-Particles (SiNPs) created via Pulsed Laser Ablation (PLA),
    we explore the dynamics of PLA in a magnetic field to study its effects on particle displacement. 
    The aim is to design filtration processes using an external magnetic field to influence SiNP size distributions, 
    given the expected charges on the particles due to the PLA process and their interactions with the magnetic field.

    The source code can be found on the [Nano Particle Playground GitHub repo](https://github.com/williammcmp/Nano-Particle-Playground)
    '''

    return intro


def expermentalMainText():
    mainText = f'''
    ## Simulation and Experimental Results 

    Experminental data was collected using images captured on a Scanning Electron Microscope 
    ([SEM](https://en.wikipedia.org/wiki/Scanning_electron_microscope)) and analysised using [imageJ](https://imagej.nih.gov/ij/), an image analysis tool.

    Explore the options below to visualize how the magnetic field affects the behavior of Silicon nanoparticles. 
    The simulation will automatically match the selected series of experimental data with the appropriate magnetic field.
    '''

    return mainText

def simText():
    text = f'''
    ## Simulation Figures

    The following plots represent the outcomes obtained through a simulation. 
    These visualizations provide valuable insights and data analysis resulting from our simulated experiments and computations.

    Due to the scale of the particles simulated, Brownian motion must be accounted for.

    Below is a summary of the simulation's parameters and Forces used.
    '''
    return text